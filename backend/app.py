# backend/app.py
"""
Flask Backend API for Structural Analysis
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.beam_solver import BeamSolver, SupportType, LoadType, Load
from analysis.frame_solver import FrameSolver, AnalysisType
from analysis.bs8110 import BS8110Designer

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze/beam', methods=['POST'])
def analyze_beam():
    """Analyze beam and return results"""
    try:
        data = request.json
        
        # Create solver
        solver = BeamSolver()
        
        # Add nodes
        nodes = {}
        for node_data in data['nodes']:
            node_id = solver.add_node(
                x=node_data['x'],
                support=SupportType(node_data['support']) if node_data['support'] else None
            )
            nodes[node_data['id']] = node_id
        
        # Add elements
        for elem_data in data['elements']:
            solver.add_element(
                node_i=nodes[elem_data['node_i']],
                node_j=nodes[elem_data['node_j']],
                E=elem_data.get('E', 200e9),
                I=elem_data.get('I', 1e-4),
                A=elem_data.get('A', 1e-2)
            )
        
        # Add loads
        for load_data in data['loads']:
            load = Load(
                type=LoadType(load_data['type']),
                element_id=load_data['element_id'],
                magnitude=load_data['magnitude'],
                position=load_data.get('position', 0),
                end_magnitude=load_data.get('end_magnitude', 0)
            )
            solver.add_load(load)

        # Add settlements (if any). The payload may include node settlements.
        for node_data in data['nodes']:
            if 'settlement' in node_data and node_data['settlement'] != 0:
                solver.add_settlement(nodes[node_data['id']], node_data['settlement'])
        
        # Solve
        solver.solve()
        
        # Get results
        internal_forces = solver.get_internal_forces(n_points=50)
        reactions = solver.get_reactions_table()
        
        # Generate plots
        plots = generate_beam_plots(internal_forces, data)
        
        return jsonify({
            'success': True,
            'internal_forces': {
                'x': internal_forces['x'],
                # convention: positive shear downward, positive moment sagging
                'moment': [-m/1000 for m in internal_forces['moment']],  # Convert to kNm
                'shear': [-v/1000 for v in internal_forces['shear']],    # Convert to kN
                'deflection': internal_forces['deflection']
            },
            'reactions': reactions,
            'plots': plots,
            'summary': {
                'max_moment': max(abs(m) for m in internal_forces['moment']) / 1000,
                'max_shear': max(abs(v) for v in internal_forces['shear']) / 1000,
                'max_deflection': max(abs(d) for d in internal_forces['deflection'])
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analyze/frame', methods=['POST'])
def analyze_frame():
    """Analyze 2D frame"""
    try:
        data = request.json
        
        analysis_type = AnalysisType(data.get('analysis_type', 'sway'))
        solver = FrameSolver(analysis_type=analysis_type)
        
        # Add nodes
        nodes = {}
        for node_data in data['nodes']:
            node_id = solver.add_node(
                x=node_data['x'],
                y=node_data['y'],
                support=node_data.get('support', 'free')
            )
            nodes[node_data['id']] = node_id
        
        # Add elements
        for elem_data in data['elements']:
            solver.add_element(
                node_i=nodes[elem_data['node_i']],
                node_j=nodes[elem_data['node_j']],
                E=elem_data.get('E', 200e9),
                I=elem_data.get('I', 1e-4),
                A=elem_data.get('A', 1e-2)
            )
        
        # Add loads
        for load_data in data.get('loads', []):
            solver.add_load(load_data['type'], **load_data['params'])
        
        # Solve
        solver.solve()
        
        # Get results
        member_forces = solver.get_member_forces()
        nodal_results = solver.get_nodal_results()
        
        # Generate frame plot
        plot = generate_frame_plot(solver, data)
        
        return jsonify({
            'success': True,
            'member_forces': member_forces,
            'nodal_results': nodal_results,
            'plot': plot,
            'analysis_type': analysis_type.value
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/design/bs8110', methods=['POST'])
def design_bs8110():
    """Design reinforcement to BS8110"""
    try:
        data = request.json
        
        designer = BS8110Designer()
        
        # Override defaults if provided
        if 'fcu' in data:
            designer.fcu = data['fcu']
        if 'fy' in data:
            designer.fy = data['fy']
        
        results = designer.design_rectangular_section(
            breadth=data['breadth'],
            depth=data['depth'],
            moment=data['moment'],
            shear=data['shear'],
            is_sagging=data.get('is_sagging', True)
        )
        
        summary = designer.get_design_summary(results)
        
        return jsonify({
            'success': True,
            'design': {
                'main_reinforcement': results.num_bars,
                'bar_diameter': results.bar_diameter,
                'area_required_mm2': round(results.area_required, 2),
                'area_provided_mm2': round(results.area_provided, 2),
                'moment_capacity_kNm': round(results.capacity, 2),
                'utilization_percent': round(results.utilization * 100, 1),
                'crack_width_mm': round(results.crack_width, 3),
                'shear_links': results.links_required,
                'link_diameter': results.link_diameter,
                'link_spacing': round(results.link_spacing, 0),
                'spacing_mm': round(results.spacing, 1)
            },
            'summary': summary,
            'bs8110_params': {
                'concrete_grade': f"C{designer.fcu}",
                'steel_grade': f"{designer.fy}",
                'cover_mm': designer.cover
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def generate_beam_plots(forces, data):
    """Generate matplotlib plots and return as base64"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    x = forces['x']
    
    # Shear diagram (apply sign convention used in API response)
    ax = axes[0]
    shear_vals = -np.array(forces['shear'])/1000  # positive downward
    ax.fill_between(x, 0, shear_vals, alpha=0.3, color='blue')
    ax.plot(x, shear_vals, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylabel('Shear (kN)')
    ax.set_title('Shear Force Diagram')
    ax.grid(True, alpha=0.3)

    # Moment diagram (positive sagging)
    ax = axes[1]
    moment_knm = -np.array(forces['moment'])/1000
    ax.fill_between(x, 0, moment_knm, alpha=0.3, color='green')
    ax.plot(x, moment_knm, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylabel('Moment (kNm)')
    ax.set_title('Bending Moment Diagram')
    ax.grid(True, alpha=0.3)
    
    # Deflection
    ax = axes[2]
    ax.fill_between(x, 0, forces['deflection'], alpha=0.3, color='purple')
    ax.plot(x, forces['deflection'], 'm-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylabel('Deflection (m)')
    ax.set_xlabel('Position (m)')
    ax.set_title('Deflection')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def generate_frame_plot(solver, data):
    """Generate frame deformation plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot undeformed shape
    for elem in solver.elements:
        node_i = solver.nodes[elem.node_i]
        node_j = solver.nodes[elem.node_j]
        ax.plot([node_i.x, node_j.x], [node_i.y, node_j.y], 
                'k--', alpha=0.3, linewidth=1)
    
    # Plot deformed shape (scaled)
    scale = 10
    for elem in solver.elements:
        node_i = solver.nodes[elem.node_i]
        node_j = solver.nodes[elem.node_j]
        
        di = solver.displacements[elem.node_i*3:elem.node_i*3+2]
        dj = solver.displacements[elem.node_j*3:elem.node_j*3+2]
        
        ax.plot([node_i.x + scale*di[0], node_j.x + scale*dj[0]],
                [node_i.y + scale*di[1], node_j.y + scale*dj[1]],
                'b-', linewidth=2)
    
    ax.set_aspect('equal')
    ax.set_title(f'Frame Analysis ({solver.analysis_type.value})')
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

if __name__ == '__main__':
    app.run(debug=True, port=5000)