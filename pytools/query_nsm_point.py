"""
Query Nearest Point on NSM Model using Nagata Patches
"""

import argparse
import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

try:
    from nsm_reader import load_nsm
    from nagata_patch import NagataModelQuery
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Query nearest point and normal on NSM model using Nagata Patch interpolation')
    parser.add_argument('filepath', help='Path to NSM file')
    parser.add_argument('x', type=float, help='Query Point X')
    parser.add_argument('y', type=float, help='Query Point Y')
    parser.add_argument('z', type=float, help='Query Point Z')
    parser.add_argument('-k', '--k-nearest', type=int, default=16, help='Number of candidate patches to check (default: 16)')
    parser.add_argument('--vis', action='store_true', help='Enable interactive visualization with PyVista')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
        
    # Load Model
    print(f"Loading model: {args.filepath}...")
    try:
        mesh_data = load_nsm(args.filepath)
    except Exception as e:
        print(f"Failed to load NSM: {e}")
        sys.exit(1)
        
    # Initialize Query Engine
    print("Initializing Nagata Query Engine (building index)...")
    model = NagataModelQuery(
        vertices=mesh_data.vertices,
        triangles=mesh_data.triangles,
        tri_vertex_normals=mesh_data.tri_vertex_normals
    )
    
    # Query
    query_pt = np.array([args.x, args.y, args.z])
    print(f"\nQuerying point: {query_pt}")
    
    result = model.query(query_pt, k_nearest=args.k_nearest)
    
    if result:
        print("-" * 40)
        print("Result Found:")
        print(f"  Nearest Point:   [{result['nearest_point'][0]:.6f}, {result['nearest_point'][1]:.6f}, {result['nearest_point'][2]:.6f}]")
        print(f"  Distance:        {result['distance']:.6f}")
        print(f"  Surface Normal:  [{result['normal'][0]:.6f}, {result['normal'][1]:.6f}, {result['normal'][2]:.6f}]")
        print(f"  Patch Index:     {result['triangle_index']}")
        print(f"  UV Parameters:   u={result['uv'][0]:.4f}, v={result['uv'][1]:.4f}")
        
        # Calculate signed distance sign
        diff = query_pt - result['nearest_point']
        sign = 1.0 if np.dot(diff, result['normal']) >= 0 else -1.0
        print(f"  Signed Distance: {sign * result['distance']:.6f}")
        print("-" * 40)
        
        # Visualization
        if args.vis:
            try:
                import pyvista as pv
                print("\nStarting visualization...")
                
                plotter = pv.Plotter()
                
                # 1. Add Mesh (Wireframe or Surface)
                # Create PyVista mesh from NSM data
                # Identify triangles
                faces = np.hstack([[3, tri[0], tri[1], tri[2]] for tri in mesh_data.triangles])
                surf = pv.PolyData(mesh_data.vertices, faces)
                
                plotter.add_mesh(surf, color='white', style='wireframe', opacity=0.3, label='NSM Wireframe')
                plotter.add_mesh(surf, color='lightblue', opacity=0.1, label='NSM Surface')
                
                # 2. Add Query Point (Red Sphere)
                plotter.add_mesh(
                    pv.Sphere(radius=0.05, center=query_pt), 
                    color='red', label='Query Point'
                )
                
                # 3. Add Nearest Point (Green Sphere)
                nearest_pt = result['nearest_point']
                plotter.add_mesh(
                    pv.Sphere(radius=0.05, center=nearest_pt), 
                    color='green', label='Nearest Point'
                )
                
                # 4. Add Normal Vector (Yellow Arrow) at Query Point
                # Shows the gradient direction at the query point
                normal = result['normal']
                arrow = pv.Arrow(start=query_pt, direction=normal, scale=0.5)
                plotter.add_mesh(arrow, color='yellow', label='SDF Gradient')
                
                # 5. Add Connection Line (Query -> Nearest)
                line = pv.Line(query_pt, nearest_pt)
                plotter.add_mesh(line, color='blue', line_width=3, label='Distance')
                
                # Set Legend Font to Times New Roman
                plotter.add_legend(font_family='times', bcolor=(0.1, 0.1, 0.1), border=True)
                plotter.add_axes()
                plotter.show()
                
            except ImportError:
                print("Error: PyVista not installed. Install with `pip install pyvista` to use --vis.")
    else:
        print("Error: Query failed (no result found).")

if __name__ == '__main__':
    main()
