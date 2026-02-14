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
    parser.add_argument('-n', '--num-points', type=int, default=0, help='Number of random query points (0 disables batch mode)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for batch mode')
    parser.add_argument('--bbox-pad', type=float, default=0.05, help='Padding ratio for mesh bounding box in batch mode')
    parser.add_argument('--vis-lines', type=int, default=200, help='Max number of query-to-nearest lines to draw in batch mode')
    
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
        tri_vertex_normals=mesh_data.tri_vertex_normals,
        nsm_filepath=args.filepath
    )
    
    # Query
    query_pt = np.array([args.x, args.y, args.z])
    def run_query(pt: np.ndarray):
        res = model.query_feature_aware(pt, k_nearest=args.k_nearest)
        if not res:
            res = model.query(pt, k_nearest=args.k_nearest)
        if not res:
            return None

        diff = pt - res['nearest_point']
        dot_val = float(np.dot(diff, res['normal']))
        dist_geo = float(np.linalg.norm(diff))
        eps_ambiguous = 1e-6
        sign = 0.0 if abs(dot_val) < eps_ambiguous * max(1.0, dist_geo) else (1.0 if dot_val >= 0.0 else -1.0)
        signed_distance = float(sign * res['distance'])

        out = dict(res)
        out['query_point'] = pt
        out['signed_distance'] = signed_distance
        out['dot_val'] = dot_val
        out['dist_geo'] = dist_geo
        return out

    if args.num_points and args.num_points > 0:
        rng = np.random.default_rng(int(args.seed))
        vmin = np.min(mesh_data.vertices, axis=0)
        vmax = np.max(mesh_data.vertices, axis=0)
        size = vmax - vmin
        pad = float(args.bbox_pad)
        vmin = vmin - pad * size
        vmax = vmax + pad * size

        pts = rng.uniform(vmin, vmax, size=(int(args.num_points), 3))
        print(f"\nBatch querying {pts.shape[0]} points in AABB:")
        print(f"  min: [{vmin[0]:.6f}, {vmin[1]:.6f}, {vmin[2]:.6f}]")
        print(f"  max: [{vmax[0]:.6f}, {vmax[1]:.6f}, {vmax[2]:.6f}]")

        results = []
        feature_counts = {}
        for i in range(pts.shape[0]):
            r = run_query(pts[i])
            if not r:
                continue
            results.append(r)
            ft = r.get('feature_type', 'UNKNOWN')
            feature_counts[ft] = feature_counts.get(ft, 0) + 1

        if not results:
            print("Error: Batch query failed (no result found).")
            return

        signed_dists = np.array([r['signed_distance'] for r in results], dtype=float)
        dists = np.array([r['distance'] for r in results], dtype=float)

        print("-" * 40)
        print("Batch Summary:")
        print(f"  Results:         {len(results)}/{pts.shape[0]}")
        print(f"  Distance min/max:{float(np.min(dists)):.6f} / {float(np.max(dists)):.6f}")
        print(f"  Signed min/max:  {float(np.min(signed_dists)):.6f} / {float(np.max(signed_dists)):.6f}")
        for k in sorted(feature_counts.keys()):
            print(f"  {k:12s}: {feature_counts[k]}")
        print("-" * 40)

        if args.vis:
            try:
                import pyvista as pv
                print("\nStarting visualization...")

                from nagata_storage import get_eng_filepath, has_cached_data, load_enhanced_data
                from visualize_nagata import create_nagata_mesh, create_nagata_mesh_enhanced

                plotter = pv.Plotter()
                if has_cached_data(args.filepath):
                    eng_path = get_eng_filepath(args.filepath)
                    cached_c_sharps = load_enhanced_data(eng_path)
                    nagata_mesh, _ = create_nagata_mesh_enhanced(
                        mesh_data.vertices,
                        mesh_data.triangles,
                        mesh_data.tri_vertex_normals,
                        mesh_data.tri_face_ids,
                        10,
                        cached_c_sharps=cached_c_sharps
                    )
                else:
                    nagata_mesh = create_nagata_mesh(
                        mesh_data.vertices,
                        mesh_data.triangles,
                        mesh_data.tri_vertex_normals,
                        mesh_data.tri_face_ids,
                        10
                    )
                if nagata_mesh.n_points > 0:
                    plotter.add_mesh(nagata_mesh, color='lightblue', opacity=0.15)

                q_points = np.stack([r['query_point'] for r in results], axis=0)
                q_poly = pv.PolyData(q_points)
                q_poly['sdf'] = signed_dists
                plotter.add_mesh(q_poly, scalars='sdf', cmap='coolwarm', point_size=6, render_points_as_spheres=True)

                nearest_points = np.stack([r['nearest_point'] for r in results], axis=0)
                n_poly = pv.PolyData(nearest_points)
                plotter.add_mesh(n_poly, color='green', point_size=4, render_points_as_spheres=True)

                num_lines = min(int(args.vis_lines), len(results))
                for i in range(num_lines):
                    line = pv.Line(results[i]['query_point'], results[i]['nearest_point'])
                    plotter.add_mesh(line, color='gray', line_width=1)

                plotter.add_axes()
                plotter.show()
            except ImportError:
                print("Error: PyVista not installed. Install with `pip install pyvista` to use --vis.")
        return

    print(f"\nQuerying point: {query_pt}")
    result = run_query(query_pt)

    if result:
        print("-" * 40)
        print("Result Found:")
        print(f"  Nearest Point:   [{result['nearest_point'][0]:.6f}, {result['nearest_point'][1]:.6f}, {result['nearest_point'][2]:.6f}]")
        print(f"  Distance:        {result['distance']:.6f}")
        print(f"  Surface Normal:  [{result['normal'][0]:.6f}, {result['normal'][1]:.6f}, {result['normal'][2]:.6f}]")
        print(f"  Patch Index:     {result['triangle_index']}")
        print(f"  UV Parameters:   u={result['uv'][0]:.4f}, v={result['uv'][1]:.4f}")
        if 'feature_type' in result:
            print(f"  Feature Type:    {result['feature_type']}")
        print(f"  Signed Distance: {result['signed_distance']:.6f}")
        print("-" * 40)

        if args.vis:
            try:
                import pyvista as pv
                print("\nStarting visualization...")

                from nagata_storage import get_eng_filepath, has_cached_data, load_enhanced_data
                from visualize_nagata import create_nagata_mesh, create_nagata_mesh_enhanced

                plotter = pv.Plotter()
                if has_cached_data(args.filepath):
                    eng_path = get_eng_filepath(args.filepath)
                    cached_c_sharps = load_enhanced_data(eng_path)
                    nagata_mesh, _ = create_nagata_mesh_enhanced(
                        mesh_data.vertices,
                        mesh_data.triangles,
                        mesh_data.tri_vertex_normals,
                        mesh_data.tri_face_ids,
                        10,
                        cached_c_sharps=cached_c_sharps
                    )
                else:
                    nagata_mesh = create_nagata_mesh(
                        mesh_data.vertices,
                        mesh_data.triangles,
                        mesh_data.tri_vertex_normals,
                        mesh_data.tri_face_ids,
                        10
                    )
                if nagata_mesh.n_points > 0:
                    plotter.add_mesh(nagata_mesh, color='lightblue', opacity=0.2)

                plotter.add_mesh(pv.Sphere(radius=0.05, center=query_pt), color='red', label='Query Point')
                nearest_pt = result['nearest_point']
                plotter.add_mesh(pv.Sphere(radius=0.05, center=nearest_pt), color='green', label='Nearest Point')

                normal = result['normal']
                arrow = pv.Arrow(start=query_pt, direction=normal, scale=0.5)
                plotter.add_mesh(arrow, color='yellow', label='SDF Gradient')

                line = pv.Line(query_pt, nearest_pt)
                plotter.add_mesh(line, color='blue', line_width=3, label='Distance')

                plotter.add_legend(font_family='times', bcolor=(0.1, 0.1, 0.1), border=True)
                plotter.add_axes()
                plotter.show()
            except ImportError:
                print("Error: PyVista not installed. Install with `pip install pyvista` to use --vis.")
    else:
        print("Error: Query failed (no result found).")

if __name__ == '__main__':
    main()
