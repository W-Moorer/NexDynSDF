import argparse
import math
import subprocess
from pathlib import Path
import numpy as np

from nagata_exporter import export_enhanced_nagata_subdivision


def _load_obj(path: Path):
    vertices = []
    normals = []
    faces = []
    face_normals = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vn "):
                parts = line.strip().split()
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                items = line.strip().split()[1:]
                face = []
                vn_idx = []
                for item in items:
                    parts = item.split("/")
                    v_idx = int(parts[0]) - 1
                    face.append(v_idx)
                    if len(parts) >= 3 and parts[2] != "":
                        vn_idx.append(int(parts[2]) - 1)
                faces.append(face)
                if vn_idx:
                    face_normals.append(vn_idx)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    normals = np.asarray(normals, dtype=np.float64) if normals else None
    return vertices, normals, faces


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray):
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
    norms[norms < 1e-12] = 1.0
    normals = normals / norms[:, None]
    return normals


def _compare_meshes(obj_a: Path, obj_b: Path, pos_tol: float, normal_tol_deg: float):
    v_a, n_a, f_a = _load_obj(obj_a)
    v_b, n_b, f_b = _load_obj(obj_b)

    result = {
        "vertex_count_a": len(v_a),
        "vertex_count_b": len(v_b),
        "face_count_a": len(f_a),
        "face_count_b": len(f_b),
        "topology_equal": False,
        "pos_max": None,
        "pos_mean": None,
        "normal_max_deg": None,
        "normal_mean_deg": None,
        "pass": False
    }

    if len(f_a) == len(f_b) and np.array_equal(f_a, f_b):
        result["topology_equal"] = True

    if len(v_a) != len(v_b):
        return result

    diff = v_a - v_b
    dist = np.linalg.norm(diff, axis=1)
    result["pos_max"] = float(np.max(dist)) if len(dist) else 0.0
    result["pos_mean"] = float(np.mean(dist)) if len(dist) else 0.0

    if n_a is None:
        n_a = _compute_vertex_normals(v_a, f_a)
    if n_b is None:
        n_b = _compute_vertex_normals(v_b, f_b)

    dot = np.sum(n_a * n_b, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    result["normal_max_deg"] = float(np.max(ang)) if len(ang) else 0.0
    result["normal_mean_deg"] = float(np.mean(ang)) if len(ang) else 0.0

    result["pass"] = (
        result["topology_equal"] and
        result["pos_max"] is not None and result["pos_max"] <= pos_tol and
        result["normal_max_deg"] is not None and result["normal_max_deg"] <= normal_tol_deg
    )
    return result


def _run_cpp_exporter(exe_path: Path, nsm_path: Path, output_dir: Path, level: int, tolerance: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(exe_path), str(nsm_path), str(output_dir), str(level), str(tolerance)],
        check=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsm", required=True)
    parser.add_argument("--cpp", required=True)
    parser.add_argument("--output")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--pos-tol", type=float, default=2e-5)
    parser.add_argument("--normal-tol", type=float, default=0.5)
    args = parser.parse_args()

    nsm_path = Path(args.nsm)
    cpp_exe = Path(args.cpp)
    if args.output:
        output_dir = Path(args.output)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "output" / "nagata_validation"
    stem = nsm_path.stem

    cpp_dir = output_dir / "cpp"
    py_dir = output_dir / "python"

    all_pass = True

    for level in args.levels:
        _run_cpp_exporter(cpp_exe, nsm_path, cpp_dir, level, args.tolerance)
        export_enhanced_nagata_subdivision(str(nsm_path), str(py_dir), level, args.tolerance)

        cpp_obj = cpp_dir / f"{stem}_enhanced_L{level}.obj"
        py_obj = py_dir / f"{stem}_enhanced_L{level}.obj"
        result = _compare_meshes(cpp_obj, py_obj, args.pos_tol, args.normal_tol)

        print(f"L{level} vertex {result['vertex_count_a']} vs {result['vertex_count_b']}")
        print(f"L{level} face {result['face_count_a']} vs {result['face_count_b']}")
        print(f"L{level} topology_equal {result['topology_equal']}")
        print(f"L{level} pos_max {result['pos_max']} pos_mean {result['pos_mean']}")
        print(f"L{level} normal_max_deg {result['normal_max_deg']} normal_mean_deg {result['normal_mean_deg']}")
        print(f"L{level} pass {result['pass']}")

        if not result["pass"]:
            all_pass = False

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
