# backend/analysis/beam_solver.py
"""
Beam Analysis Solver - Matrix Stiffness Method
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

class SupportType(Enum):
    FIXED = "fixed"
    PINNED = "pinned"
    ROLLER = "roller"
    FREE = "free"

class LoadType(Enum):
    POINT = "point"
    UDL = "udl"
    TRIANGULAR = "triangular"
    MOMENT = "moment"

@dataclass
class Node:
    id: int
    x: float
    y: float = 0.0
    support: Optional[SupportType] = None
    fx: float = 0.0  # Reaction force x
    fy: float = 0.0  # Reaction force y
    mz: float = 0.0  # Reaction moment
    
@dataclass
class Element:
    id: int
    node_i: int
    node_j: int
    E: float = 200e9  # Young's modulus (Pa)
    I: float = 1e-4   # Moment of inertia (m⁴)
    A: float = 1e-2   # Area (m²)
    
@dataclass
class Load:
    type: LoadType
    element_id: int
    magnitude: float  # N or N/m
    position: float = 0.0  # For point loads, distance from node_i
    end_magnitude: float = 0.0  # For triangular loads

class BeamSolver:
    def __init__(self):
        self.nodes: List[Node] = []
        self.elements: List[Element] = []
        self.loads: List[Load] = []
        self.K_global = None
        self.F_global = None
        self.displacements = None
        
    def add_node(self, x: float, support: Optional[SupportType] = None) -> int:
        """Add node and return its ID"""
        node_id = len(self.nodes)
        self.nodes.append(Node(id=node_id, x=x, support=support))
        return node_id
    
    def add_element(self, node_i: int, node_j: int, 
                    E: float = 200e9, I: float = 1e-4, A: float = 1e-2) -> int:
        """Add beam element"""
        elem_id = len(self.elements)
        self.elements.append(Element(
            id=elem_id, node_i=node_i, node_j=node_j, E=E, I=I, A=A
        ))
        return elem_id
    
    def add_load(self, load: Load):
        """Add load to beam"""
        self.loads.append(load)
    
    def _get_element_length(self, elem: Element) -> float:
        """Calculate element length"""
        node_i = self.nodes[elem.node_i]
        node_j = self.nodes[elem.node_j]
        return np.sqrt((node_j.x - node_i.x)**2 + (node_j.y - node_i.y)**2)
    
    def _get_element_stiffness(self, elem: Element) -> np.ndarray:
        """Calculate local element stiffness matrix (beam, 4 DOF)"""
        L = self._get_element_length(elem)
        E = elem.E
        I = elem.I
        
        # Beam stiffness matrix (4x4: v_i, θ_i, v_j, θ_j)
        k = np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ]) * E * I / L**3
        
        return k
    
    def _assemble_global_stiffness(self):
        """Assemble global stiffness matrix"""
        n_nodes = len(self.nodes)
        n_dof = n_nodes * 2  # 2 DOF per node (v, θ)
        
        K = np.zeros((n_dof, n_dof))
        
        for elem in self.elements:
            k_local = self._get_element_stiffness(elem)
            
            # Global DOF indices
            dof_i = elem.node_i * 2
            dof_j = elem.node_j * 2
            
            # Assemble into global matrix
            dof_map = [dof_i, dof_i+1, dof_j, dof_j+1]
            
            for i in range(4):
                for j in range(4):
                    K[dof_map[i], dof_map[j]] += k_local[i, j]
        
        self.K_global = K
        return K
    
    def _apply_boundary_conditions(self):
        """Apply support boundary conditions"""
        n_nodes = len(self.nodes)
        
        # Identify constrained DOF
        constrained = []
        
        for node in self.nodes:
            if node.support == SupportType.FIXED:
                constrained.extend([node.id*2, node.id*2+1])  # v and θ
            elif node.support in [SupportType.PINNED, SupportType.ROLLER]:
                constrained.append(node.id*2)  # v only
        
        self.constrained_dof = constrained
        self.free_dof = [i for i in range(n_nodes*2) if i not in constrained]
        
        return constrained
    
    def _assemble_load_vector(self):
        """Assemble global load vector including fixed-end forces"""
        n_nodes = len(self.nodes)
        F = np.zeros(n_nodes * 2)
        
        for load in self.loads:
            elem = self.elements[load.element_id]
            L = self._get_element_length(elem)
            
            if load.type == LoadType.POINT:
                # Point load fixed-end forces
                a = load.position
                b = L - a
                P = load.magnitude
                
                # Vertical forces and moments at each end
                f_i = P * b**2 * (3*a + b) / L**3
                m_i = P * a * b**2 / L**2
                f_j = P * a**2 * (a + 3*b) / L**3
                m_j = -P * a**2 * b / L**2
                
            elif load.type == LoadType.UDL:
                w = load.magnitude
                f_i = f_j = w * L / 2
                m_i = w * L**2 / 12
                m_j = -w * L**2 / 12
                
            else:
                continue
            
            # Add to global vector
            dof_i = elem.node_i * 2
            dof_j = elem.node_j * 2
            
            F[dof_i] += f_i
            F[dof_i+1] += m_i
            F[dof_j] += f_j
            F[dof_j+1] += m_j
        
        self.F_global = F
        return F
    
    def solve(self):
        """Solve the beam system"""
        # Assemble matrices
        K = self._assemble_global_stiffness()
        F = self._assemble_load_vector()
        constrained = self._apply_boundary_conditions()
        
        # Reduce system
        K_reduced = K[np.ix_(self.free_dof, self.free_dof)]
        F_reduced = F[self.free_dof]
        
        # Solve for displacements
        u_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Full displacement vector
        u = np.zeros(len(self.nodes) * 2)
        u[self.free_dof] = u_reduced
        
        self.displacements = u
        
        # Calculate reactions
        reactions = K @ u - F
        self.reactions = reactions
        
        # Store reactions in nodes
        for node in self.nodes:
            node.fy = reactions[node.id*2]
            node.mz = reactions[node.id*2+1]
        
        return u
    
    def get_internal_forces(self, n_points: int = 20) -> Dict:
        """Calculate internal forces along beam using equilibrium rather than
        nodal displacement derivatives.  Supports point loads and element-wide
        UDLs; partial UDLs are approximated by element splitting upstream.
        """
        if self.displacements is None:
            self.solve()

        # helper to interpolate deflection using shape functions (previous
        # implementation) so we still return something useful
        def interpolate_deflection(x_global):
            # find containing element
            for elem in self.elements:
                node_i = self.nodes[elem.node_i]
                node_j = self.nodes[elem.node_j]
                if node_i.x - 1e-8 <= x_global <= node_j.x + 1e-8:
                    L = self._get_element_length(elem)
                    xi = (x_global - node_i.x) / (node_j.x - node_i.x)
                    u_elem = np.array([
                        self.displacements[elem.node_i*2],
                        self.displacements[elem.node_i*2+1],
                        self.displacements[elem.node_j*2],
                        self.displacements[elem.node_j*2+1]
                    ])
                    N = np.array([
                        1 - 3*xi**2 + 2*xi**3,
                        L*(xi - 2*xi**2 + xi**3),
                        3*xi**2 - 2*xi**3,
                        L*(-xi**2 + xi**3)
                    ])
                    return N @ u_elem
            return 0.0

        # gather sample locations evenly across total length
        total_length = max(n.x for n in self.nodes) if self.nodes else 0
        x_samples = np.linspace(0, total_length, n_points)

        results = {'x': [], 'moment': [], 'shear': [], 'deflection': []}

        # precompute global loads for convenience
        global_loads = []  # list of (type, global_position, magnitude, end_pos?)
        for load in self.loads:
            elem = self.elements[load.element_id]
            start = self.nodes[elem.node_i].x
            if load.type == LoadType.POINT:
                global_loads.append(('point', start + load.position, load.magnitude))
            elif load.type == LoadType.UDL:
                # assume covers full element (we split upstream for partials)
                end = self.nodes[elem.node_j].x
                global_loads.append(('udl', start, load.magnitude, end))
            # other load types could be added here

        # compute shear/moment at each sample via sectional equilibrium
        for x in x_samples:
            # shear: sum of reactions left of x minus applied loads left of x
            V = 0.0
            for node in self.nodes:
                if node.x <= x and abs(node.fy) > 1e-12:
                    V += node.fy
            for gl in global_loads:
                if gl[0] == 'point':
                    _, pos, mag = gl
                    if pos <= x:
                        V -= mag
                elif gl[0] == 'udl':
                    _, s, w, e = gl
                    a = max(0.0, min(x, e) - s)
                    V -= w * a
            results['shear'].append(V)

            # moment: reactions * lever arms minus load contributions
            M = 0.0
            for node in self.nodes:
                if node.x <= x:
                    M += node.fy * (x - node.x)
            for gl in global_loads:
                if gl[0] == 'point':
                    _, pos, mag = gl
                    if pos <= x:
                        M -= mag * (x - pos)
                elif gl[0] == 'udl':
                    _, s, w, e = gl
                    a = max(0.0, min(x, e) - s)
                    if a > 0:
                        # equivalent resultant acts at mid‑span of loaded portion
                        M -= w * a * (x - (s + a/2))
            results['moment'].append(M)

            # deflection via interpolation
            results['deflection'].append(interpolate_deflection(x))
            results['x'].append(x)

        return results
    
    def get_reactions_table(self) -> List[Dict]:
        """Format reactions for display"""
        table = []
        for node in self.nodes:
            if node.support:
                table.append({
                    'node': node.id,
                    'position': f"{node.x:.3f} m",
                    'type': node.support.value,
                    'vertical_force': f"{node.fy:.3f} kN",
                    'moment': f"{node.mz:.3f} kNm"
                })
        return table