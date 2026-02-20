import argparse
from pathlib import Path
import numpy as np

from nsm_reader import load_nsm
from nagata_storage import get_eng_filepath, load_enhanced_data
from visualize_nagata import (
    detect_crease_edges,
    compute_c_sharps_for_edges,
    create_nagata_mesh_enhanced,
    _polydata_to_triangles
)


def _resolution_for_level(level: int) -> int:
    if level < 0:
        level = 0
    return (1 << level) + 1


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len < 1e-12:
            continue
        n = n / n_len
        normals[f[0]] += n
        normals[f[1]] += n
        normals[f[2]] += n
    norms = np.linalg.norm(normals, axis=1)
    mask = norms < 1e-12
    if np.any(mask):
        normals[mask] = fallback
        norms[mask] = np.linalg.norm(normals[mask], axis=1)
    normals = normals / norms[:, None]
    return normals


def export_enhanced_nagata_subdivision(
    input_path: str,
    output_dir: str,
    subdivision_level: int,
    tolerance: float,
    *,
    use_cache: bool = True,
    dtype: np.dtype = np.float64
):
    mesh_data = load_nsm(input_path)
    vertices = mesh_data.vertices.astype(dtype, copy=False)
    triangles = mesh_data.triangles
    tri_vertex_normals = mesh_data.tri_vertex_normals.astype(dtype, copy=False)

    eng_path = get_eng_filepath(input_path)
    c_sharps = load_enhanced_data(eng_path) if use_cache else None
    if c_sharps is None:
        crease_edges = detect_crease_edges(vertices, triangles, tri_vertex_normals)
        c_sharps = compute_c_sharps_for_edges(crease_edges, vertices, triangles, tri_vertex_normals, k_factor=tolerance)
    elif dtype != np.float64:
        c_sharps = {k: v.astype(dtype, copy=False) for k, v in c_sharps.items()}

    resolution = _resolution_for_level(subdivision_level)

    mesh, _ = create_nagata_mesh_enhanced(
        vertices,
        triangles,
        tri_vertex_normals,
        mesh_data.tri_face_ids,
        resolution,
        d0=tolerance,
        cached_c_sharps=c_sharps
    )
    all_vertices, all_faces = _polydata_to_triangles(mesh)
    all_vertices = all_vertices.astype(dtype, copy=False)
    all_normals = _compute_vertex_normals(
        all_vertices,
        all_faces,
        np.array([0.0, 0.0, 1.0], dtype=dtype)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    output_path = output_dir / f"{stem}_enhanced_L{subdivision_level}.obj"

    with open(output_path, "w", encoding="utf-8") as f:
        for v in all_vertices:
            f.write(f"v {v[0]:.16g} {v[1]:.16g} {v[2]:.16g}\n")
        for n in all_normals:
            f.write(f"vn {n[0]:.16g} {n[1]:.16g} {n[2]:.16g}\n")
        for face in all_faces:
            a, b, c = face
            a += 1
            b += 1
            c += 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NSM文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--levels", type=int, nargs="*")
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args()

    levels = args.levels if args.levels else [args.level]
    for level in levels:
        path = export_enhanced_nagata_subdivision(args.input, args.output_dir, level, args.tolerance)
        print(path)


if __name__ == "__main__":
    main()
