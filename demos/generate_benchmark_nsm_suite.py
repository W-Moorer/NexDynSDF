#!/usr/bin/env python3
"""
Generate a benchmark NSM model suite for Paper-1 experiments.

Reuses existing generators (sphere/cube/cone/box) and adds cylinder/torus.
Outputs:
- multiple .nsm files
- manifest txt (one model path per line)
- metadata csv (vertices/triangles)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from generate_box_nsm import generate_box_nsm
from generate_cone_nsm import generate_refined_cone_nsm
from generate_cube_nsm import generate_cube_nsm
from generate_sphere_nsm import generate_sphere_nsm


@dataclass(frozen=True)
class ModelMeta:
    name: str
    category: str
    path: Path
    vertices: int
    triangles: int
    param_json: str


def save_nsm(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_face_ids: np.ndarray,
    tri_vertex_normals: np.ndarray,
    output_path: Path,
) -> Tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.uint32)
    fids = np.asarray(tri_face_ids, dtype=np.uint32)
    norms = np.asarray(tri_vertex_normals, dtype=np.float64)

    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("triangles should be [M,3]")
    if fids.shape[0] != t.shape[0]:
        raise ValueError("tri_face_ids size mismatch")
    if norms.shape != (t.shape[0], 3, 3):
        raise ValueError("tri_vertex_normals shape mismatch")

    with output_path.open("wb") as f:
        f.write(b"NSM\x00")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", int(v.shape[0])))
        f.write(struct.pack("<I", int(t.shape[0])))
        f.write(b"\x00" * 48)
        v.tofile(f)
        t.tofile(f)
        fids.tofile(f)
        norms.tofile(f)

    return int(v.shape[0]), int(t.shape[0])


def generate_cylinder_nsm(
    radius: float,
    height: float,
    n_segments: int,
    n_height: int,
    n_cap_rings: int,
    output_path: Path,
) -> Tuple[int, int]:
    vertices: List[List[float]] = []
    triangles: List[List[int]] = []
    tri_face_ids: List[int] = []

    h2 = 0.5 * float(height)
    r = float(radius)

    # Side rings: top -> bottom
    for hi in range(n_height + 1):
        z = h2 - (2.0 * h2) * (hi / n_height)
        for si in range(n_segments):
            theta = 2.0 * np.pi * (si / n_segments)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, z])

    side_top_start = 0
    side_bottom_start = n_height * n_segments

    # Top center and bottom center
    top_center = len(vertices)
    vertices.append([0.0, 0.0, h2])
    bottom_center = len(vertices)
    vertices.append([0.0, 0.0, -h2])

    # Cap internal rings (top/bottom)
    top_ring_starts: List[int] = []
    bottom_ring_starts: List[int] = []
    for ri in range(1, n_cap_rings):
        rr = r * (ri / n_cap_rings)
        t_start = len(vertices)
        top_ring_starts.append(t_start)
        for si in range(n_segments):
            theta = 2.0 * np.pi * (si / n_segments)
            vertices.append([rr * np.cos(theta), rr * np.sin(theta), h2])
        b_start = len(vertices)
        bottom_ring_starts.append(b_start)
        for si in range(n_segments):
            theta = 2.0 * np.pi * (si / n_segments)
            vertices.append([rr * np.cos(theta), rr * np.sin(theta), -h2])

    # Side faces (face id 0)
    for hi in range(n_height):
        r0 = hi * n_segments
        r1 = (hi + 1) * n_segments
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            a = r0 + si
            b = r0 + sn
            c = r1 + si
            d = r1 + sn
            triangles.append([a, c, b])
            tri_face_ids.append(0)
            triangles.append([b, c, d])
            tri_face_ids.append(0)

    # Top cap (face id 1, +Z)
    if n_cap_rings <= 1:
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            triangles.append([top_center, side_top_start + si, side_top_start + sn])
            tri_face_ids.append(1)
    else:
        first = top_ring_starts[0]
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            triangles.append([top_center, first + si, first + sn])
            tri_face_ids.append(1)

        for ridx in range(len(top_ring_starts) - 1):
            inner = top_ring_starts[ridx]
            outer = top_ring_starts[ridx + 1]
            for si in range(n_segments):
                sn = (si + 1) % n_segments
                triangles.append([inner + si, outer + si, inner + sn])
                tri_face_ids.append(1)
                triangles.append([inner + sn, outer + si, outer + sn])
                tri_face_ids.append(1)

        last_inner = top_ring_starts[-1]
        outer = side_top_start
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            triangles.append([last_inner + si, outer + si, last_inner + sn])
            tri_face_ids.append(1)
            triangles.append([last_inner + sn, outer + si, outer + sn])
            tri_face_ids.append(1)

    # Bottom cap (face id 2, -Z)
    if n_cap_rings <= 1:
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            triangles.append([bottom_center, side_bottom_start + sn, side_bottom_start + si])
            tri_face_ids.append(2)
    else:
        first = bottom_ring_starts[0]
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            triangles.append([bottom_center, first + sn, first + si])
            tri_face_ids.append(2)

        for ridx in range(len(bottom_ring_starts) - 1):
            inner = bottom_ring_starts[ridx]
            outer = bottom_ring_starts[ridx + 1]
            for si in range(n_segments):
                sn = (si + 1) % n_segments
                triangles.append([inner + si, inner + sn, outer + si])
                tri_face_ids.append(2)
                triangles.append([inner + sn, outer + sn, outer + si])
                tri_face_ids.append(2)

        last_inner = bottom_ring_starts[-1]
        outer = side_bottom_start
        for si in range(n_segments):
            sn = (si + 1) % n_segments
            triangles.append([last_inner + si, last_inner + sn, outer + si])
            tri_face_ids.append(2)
            triangles.append([last_inner + sn, outer + sn, outer + si])
            tri_face_ids.append(2)

    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.uint32)
    fids = np.asarray(tri_face_ids, dtype=np.uint32)

    tri_vertex_normals = np.zeros((t.shape[0], 3, 3), dtype=np.float64)
    for i in range(t.shape[0]):
        fid = int(fids[i])
        for j in range(3):
            vid = int(t[i, j])
            if fid == 0:
                x, y, _ = v[vid]
                ln = np.sqrt(x * x + y * y)
                if ln < 1e-12:
                    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    n = np.array([x / ln, y / ln, 0.0], dtype=np.float64)
            elif fid == 1:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            tri_vertex_normals[i, j] = n

    return save_nsm(v, t, fids, tri_vertex_normals, output_path)


def generate_torus_nsm(
    major_radius: float,
    minor_radius: float,
    n_major: int,
    n_minor: int,
    output_path: Path,
) -> Tuple[int, int]:
    vertices = np.zeros((n_major * n_minor, 3), dtype=np.float64)
    normals = np.zeros((n_major * n_minor, 3), dtype=np.float64)

    def vid(i: int, j: int) -> int:
        return i * n_minor + j

    for i in range(n_major):
        u = 2.0 * np.pi * (i / n_major)
        cu, su = np.cos(u), np.sin(u)
        for j in range(n_minor):
            v = 2.0 * np.pi * (j / n_minor)
            cv, sv = np.cos(v), np.sin(v)
            rr = major_radius + minor_radius * cv
            x = rr * cu
            y = rr * su
            z = minor_radius * sv
            vertices[vid(i, j)] = [x, y, z]
            normals[vid(i, j)] = [cu * cv, su * cv, sv]

    triangles: List[List[int]] = []
    for i in range(n_major):
        i1 = (i + 1) % n_major
        for j in range(n_minor):
            j1 = (j + 1) % n_minor
            v00 = vid(i, j)
            v01 = vid(i, j1)
            v10 = vid(i1, j)
            v11 = vid(i1, j1)
            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])

    t = np.asarray(triangles, dtype=np.uint32)
    fids = np.zeros((t.shape[0],), dtype=np.uint32)
    tri_vertex_normals = np.zeros((t.shape[0], 3, 3), dtype=np.float64)
    for ti in range(t.shape[0]):
        for lv in range(3):
            tri_vertex_normals[ti, lv] = normals[int(t[ti, lv])]

    return save_nsm(vertices, t, fids, tri_vertex_normals, output_path)


def summarize_nsm(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(64)
        if len(head) != 64:
            raise RuntimeError(f"invalid NSM header: {path}")
        if head[0:4] != b"NSM\x00":
            raise RuntimeError(f"invalid NSM magic: {path}")
        n_vertices = struct.unpack("<I", head[8:12])[0]
        n_triangles = struct.unpack("<I", head[12:16])[0]
    return int(n_vertices), int(n_triangles)


def generate_suite(output_dir: Path, profile: str, force: bool) -> List[ModelMeta]:
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: List[ModelMeta] = []

    def maybe_skip(path: Path) -> bool:
        return path.exists() and not force

    def add_meta(name: str, cat: str, path: Path, params: Dict[str, float]) -> None:
        nv, nt = summarize_nsm(path)
        produced.append(
            ModelMeta(
                name=name,
                category=cat,
                path=path,
                vertices=nv,
                triangles=nt,
                param_json=json.dumps(params, sort_keys=True),
            )
        )

    base_cfgs: List[Tuple[str, str, Dict[str, float]]] = [
        ("sphere_ref", "sphere", {"radius": 1.0, "n_lat": 24, "n_lon": 48}),
        ("cube_ref", "cube", {"half_size": 0.5, "n_segments": 8}),
        ("cone_ref", "cone", {"radius": 1.0, "height": 2.0, "n_segments": 96, "n_height": 16, "n_radius": 16}),
        ("box_thin", "box", {"length_x": 5.0, "length_y": 5.0, "length_z": 0.5, "n_segments_xy": 8, "n_segments_z": 4}),
        ("cylinder_ref", "cylinder", {"radius": 1.0, "height": 2.0, "n_segments": 96, "n_height": 12, "n_cap_rings": 12}),
        ("torus_ref", "torus", {"major_radius": 1.2, "minor_radius": 0.4, "n_major": 96, "n_minor": 48}),
    ]
    full_extra_cfgs: List[Tuple[str, str, Dict[str, float]]] = [
        ("sphere_coarse", "sphere", {"radius": 1.0, "n_lat": 12, "n_lon": 24}),
        ("cube_coarse", "cube", {"half_size": 0.5, "n_segments": 4}),
        ("cone_slender", "cone", {"radius": 0.6, "height": 2.4, "n_segments": 96, "n_height": 16, "n_radius": 12}),
        ("box_block", "box", {"length_x": 2.0, "length_y": 1.4, "length_z": 1.0, "n_segments_xy": 8, "n_segments_z": 8}),
        ("cylinder_wide", "cylinder", {"radius": 1.4, "height": 1.0, "n_segments": 96, "n_height": 10, "n_cap_rings": 10}),
        ("torus_thin", "torus", {"major_radius": 1.6, "minor_radius": 0.2, "n_major": 112, "n_minor": 40}),
    ]
    hard_cfgs: List[Tuple[str, str, Dict[str, float]]] = [
        # Cone: hard with high slenderness h/r
        ("cone_ar16", "cone", {"radius": 1.0, "height": 1.6, "n_segments": 96, "n_height": 16, "n_radius": 16}),
        ("cone_ar32", "cone", {"radius": 1.0, "height": 3.2, "n_segments": 96, "n_height": 24, "n_radius": 12}),
        ("cone_ar50", "cone", {"radius": 1.0, "height": 5.0, "n_segments": 96, "n_height": 28, "n_radius": 10}),
        # Cylinder: hard with slenderness sweep
        ("cylinder_ar08", "cylinder", {"radius": 1.0, "height": 0.8, "n_segments": 96, "n_height": 8, "n_cap_rings": 10}),
        ("cylinder_ar20", "cylinder", {"radius": 1.0, "height": 2.0, "n_segments": 96, "n_height": 14, "n_cap_rings": 10}),
        ("cylinder_ar40", "cylinder", {"radius": 1.0, "height": 4.0, "n_segments": 96, "n_height": 24, "n_cap_rings": 10}),
        # Torus: hard with thin tube ratio r/R
        ("torus_rr25", "torus", {"major_radius": 1.6, "minor_radius": 0.40, "n_major": 112, "n_minor": 48}),
        ("torus_rr10", "torus", {"major_radius": 1.6, "minor_radius": 0.16, "n_major": 112, "n_minor": 40}),
        ("torus_rr05", "torus", {"major_radius": 1.6, "minor_radius": 0.08, "n_major": 112, "n_minor": 32}),
        # Box: hard with thin thickness ratio z/max(x,y)
        ("box_t20", "box", {"length_x": 5.0, "length_y": 5.0, "length_z": 1.0, "n_segments_xy": 10, "n_segments_z": 10}),
        ("box_t08", "box", {"length_x": 5.0, "length_y": 5.0, "length_z": 0.4, "n_segments_xy": 10, "n_segments_z": 4}),
        ("box_t02", "box", {"length_x": 5.0, "length_y": 5.0, "length_z": 0.1, "n_segments_xy": 10, "n_segments_z": 2}),
    ]

    cfgs: List[Tuple[str, str, Dict[str, float]]] = []
    if profile == "quick":
        cfgs.extend(base_cfgs)
    elif profile == "full":
        cfgs.extend(base_cfgs)
        cfgs.extend(full_extra_cfgs)
    elif profile == "hard":
        cfgs.extend(hard_cfgs)
    else:
        raise ValueError(f"unsupported profile: {profile}")

    for name, cat, params in cfgs:
        out_path = output_dir / f"{name}.nsm"
        if maybe_skip(out_path):
            add_meta(name, cat, out_path, params)
            print(f"[bench] reuse {out_path}")
            continue

        print(f"[bench] generate {name} ({cat}) -> {out_path}")
        if cat == "sphere":
            generate_sphere_nsm(
                float(params["radius"]),
                int(params["n_lat"]),
                int(params["n_lon"]),
                str(out_path),
            )
        elif cat == "cube":
            generate_cube_nsm(
                float(params["half_size"]),
                int(params["n_segments"]),
                str(out_path),
            )
        elif cat == "cone":
            generate_refined_cone_nsm(
                float(params["radius"]),
                float(params["height"]),
                int(params["n_segments"]),
                int(params["n_height"]),
                int(params["n_radius"]),
                str(out_path),
            )
        elif cat == "box":
            generate_box_nsm(
                float(params["length_x"]),
                float(params["length_y"]),
                float(params["length_z"]),
                int(params["n_segments_xy"]),
                int(params["n_segments_z"]),
                str(out_path),
            )
        elif cat == "cylinder":
            generate_cylinder_nsm(
                float(params["radius"]),
                float(params["height"]),
                int(params["n_segments"]),
                int(params["n_height"]),
                int(params["n_cap_rings"]),
                out_path,
            )
        elif cat == "torus":
            generate_torus_nsm(
                float(params["major_radius"]),
                float(params["minor_radius"]),
                int(params["n_major"]),
                int(params["n_minor"]),
                out_path,
            )
        else:
            raise ValueError(f"unsupported category: {cat}")

        add_meta(name, cat, out_path, params)

    return produced


def write_manifest(manifest_path: Path, metas: List[ModelMeta]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(m.path.resolve()) for m in metas]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata_csv(csv_path: Path, metas: List[ModelMeta]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "category",
                "path",
                "vertices",
                "triangles",
                "param_json",
            ],
        )
        writer.writeheader()
        for m in metas:
            writer.writerow(
                {
                    "name": m.name,
                    "category": m.category,
                    "path": str(m.path.resolve()),
                    "vertices": m.vertices,
                    "triangles": m.triangles,
                    "param_json": m.param_json,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate NSM benchmark suite")
    parser.add_argument("--output_dir", default="output/benchmarks/models")
    parser.add_argument("--manifest", default=None, help="output manifest txt path")
    parser.add_argument("--metadata_csv", default=None, help="output metadata csv path")
    parser.add_argument("--profile", choices=["quick", "full", "hard"], default="quick")
    parser.add_argument("--force", action="store_true", help="regenerate existing models")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    metas = generate_suite(out_dir, args.profile, args.force)

    manifest = Path(args.manifest) if args.manifest else (out_dir / "benchmark_models.txt")
    if not manifest.is_absolute():
        manifest = (repo_root / manifest).resolve()
    write_manifest(manifest, metas)

    metadata_csv = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else (out_dir / "benchmark_models_meta.csv")
    )
    if not metadata_csv.is_absolute():
        metadata_csv = (repo_root / metadata_csv).resolve()
    write_metadata_csv(metadata_csv, metas)

    print(f"[bench] generated {len(metas)} models")
    print(f"[bench] manifest: {manifest}")
    print(f"[bench] metadata: {metadata_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
