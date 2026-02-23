# backend/analysis/frame_solver.py
"""
2D Frame Analysis - Matrix Stiffness Method with Sway/Non-sway options
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class AnalysisType(Enum):
    NON_SWAY = "non_sway"  # Prevent lateral displacement
    SWAY = "sway"          # Allow lateral displacement

@dataclass
class FrameNode:
    id: int
    x: float
    y: float
    support: str = "free"  # fixed, pinned, roller, free
    restrained_dx: bool = False
    restrained_dy: bool = False
    restrained_rz: bool = False

@dataclass
class FrameElement:
    id: int
    node_i: int
    node_j: int
    E: float = 200e9
    I: float = 1e-4
    A: float = 1e-2
    
class FrameSolver:
    def __init__(self, analysis_type: AnalysisType = AnalysisType.SWAY):
        self.nodes: List[FrameNode] = []
        self.elements: List[FrameElement] = []
        self.loads: List[Dict] = []
        self.analysis_type = analysis_type
        self.K_global = None
        self.displacements = None
        
    def add_node(self, x: float, y: float, support: str = "free") -> int:
        """Add frame node"""
        node_id = len(self.nodes)
        
        # Set restraints based on support type
        restrained_dx = False
        restrained_dy = False
        restrained_rz = False
        
        if support == "fixed":
            restrained_dx = restrained_dy = restrained_rz = True
        elif support == "pinned":
            restrained_dx = restrained_dy = True
        elif support == "roller_x":
            restrained_dy = True
        elif support == "roller_y":
            restrained_dx = True
            
        # Non-sway analysis: restrain all horizontal DOF
        if self.analysis_type == AnalysisType.NON_SWAY:
            restrained_dx = True
        
        self.nodes.append(FrameNode(
            id=node_id, x=x, y=y, support=support,
            restrained_dx=restrained_dx, restrained_dy=restrained_dy,
            restrained_rz=restrained_rz
        ))
        return node_id
    
    def add_element(self, node_i: int, node_j: int,
                    E: float = 200e9, I: float = 1e-4, A: float = 1e-2) -> int:
        """Add frame element"""
        elem_id = len(self.elements)
        self.elements.append(FrameElement(
            id=elem_id, node_i=node_i, node_j=node_j, E=E, I=I, A=A
        ))
        return elem_id
    
    def add_load(self, load_type: str, **kwargs):
        """Add load to frame"""
        load = {'type': load_type, **kwargs}
        self.loads.append(load)
    
    def _get_transform_matrix(self, elem: FrameElement) -> np.ndarray:
        """Get transformation matrix for element"""
        node_i = self.nodes[elem.node_i]
        node_j = self.nodes[elem.node_j]
        
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        L = np.sqrt(dx**2 + dy**2)
        
        c = dx / L  # cos
        s = dy / L  # sin
        
        # Transformation matrix (6x6 for 2D frame)
        T = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        return T, L
    
    def _get_element_stiffness_local(self, elem: FrameElement, L: float) -> np.ndarray:
        """Local element stiffness matrix"""
        E = elem.E
        A = elem.A
        I = elem.I
        
        # 2D frame stiffness matrix (6x6)
        k = np.zeros((6, 6))
        
        # Axial terms
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 3] = -EA_L
        k[3, 0] = -EA_L
        k[3, 3] = EA_L
        
        # Flexural terms
        EI = E * I
        k[1, 1] = 12 * EI / L**3
        k[1, 2] = 6 * EI / L**2
        k[1, 4] = -12 * EI / L**3
        k[1, 5] = 6 * EI / L**2
        
        k[2, 1] = 6 * EI / L**2
        k[2, 2] = 4 * EI / L
        k[2, 4] = -6 * EI / L**2
        k[2, 5] = 2 * EI / L
        
        k[4, 1] = -12 * EI / L**3
        k[4, 2] = -6 * EI / L**2
        k[4, 4] = 12 * EI / L**3
        k[4, 5] = -6 * EI / L**2
        
        k[5, 1] = 6 * EI / L**2
        k[5, 2] = 2 * EI / L
        k[5, 4] = -6 * EI / L**2
        k[5, 5] = 4 * EI / L
        
        return k
    
    def solve(self):
        """Solve frame using matrix stiffness method"""
        n_nodes = len(self.nodes)
        n_dof = n_nodes * 3  # 3 DOF per node: u, v, θ
        
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        # Assemble global stiffness
        for elem in self.elements:
            T, L = self._get_transform_matrix(elem)
            k_local = self._get_element_stiffness_local(elem, L)
            
            # Transform to global coordinates
            k_global = T.T @ k_local @ T
            
            # Assemble
            dof_i = elem.node_i * 3
            dof_j = elem.node_j * 3
            dof_map = [dof_i, dof_i+1, dof_i+2, dof_j, dof_j+1, dof_j+2]
            
            for i in range(6):
                for j in range(6):
                    K[dof_map[i], dof_map[j]] += k_global[i, j]
        
        # Apply loads
        for load in self.loads:
            if load['type'] == 'nodal':
                node_id = load['node']
                F[node_id*3] += load.get('fx', 0)
                F[node_id*3+1] += load.get('fy', 0)
                F[node_id*3+2] += load.get('mz', 0)
        
        # Identify constrained DOF
        constrained = []
        for node in self.nodes:
            if node.restrained_dx:
                constrained.append(node.id*3)
            if node.restrained_dy:
                constrained.append(node.id*3+1)
            if node.restrained_rz:
                constrained.append(node.id*3+2)
        
        free_dof = [i for i in range(n_dof) if i not in constrained]
        
        # Solve
        K_reduced = K[np.ix_(free_dof, free_dof)]
        F_reduced = F[free_dof]
        
        u_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        u = np.zeros(n_dof)
        u[free_dof] = u_reduced
        
        self.displacements = u
        self.K_global = K
        
        # Calculate reactions
        reactions = K @ u - F
        self.reactions = reactions
        
        return u
    
    def get_member_forces(self) -> List[Dict]:
        """Calculate member end forces"""
        if self.displacements is None:
            self.solve()
        
        member_forces = []
        
        for elem in self.elements:
            T, L = self._get_transform_matrix(elem)
            
            # Get nodal displacements
            dof_i = elem.node_i * 3
            dof_j = elem.node_j * 3
            u_global = np.array([
                self.displacements[dof_i],
                self.displacements[dof_i+1],
                self.displacements[dof_i+2],
                self.displacements[dof_j],
                self.displacements[dof_j+1],
                self.displacements[dof_j+2]
            ])
            
            # Transform to local
            u_local = T @ u_global
            
            # Calculate forces
            k_local = self._get_element_stiffness_local(elem, L)
            f_local = k_local @ u_local
            
            member_forces.append({
                'element': elem.id,
                'node_i': elem.node_i,
                'node_j': elem.node_j,
                'axial_i': f_local[0],
                'shear_i': f_local[1],
                'moment_i': f_local[2],
                'axial_j': f_local[3],
                'shear_j': f_local[4],
                'moment_j': f_local[5]
            })
        
        return member_forces
    
    def get_nodal_results(self) -> List[Dict]:
        """Get nodal displacements and reactions"""
        results = []
        for node in self.nodes:
            results.append({
                'node': node.id,
                'x': node.x,
                'y': node.y,
                'displacement_x': self.displacements[node.id*3],
                'displacement_y': self.displacements[node.id*3+1],
                'rotation': self.displacements[node.id*3+2],
                'reaction_x': self.reactions[node.id*3] if node.restrained_dx else 0,
                'reaction_y': self.reactions[node.id*3+1] if node.restrained_dy else 0,
                'reaction_moment': self.reactions[node.id*3+2] if node.restrained_rz else 0
            })
        return results