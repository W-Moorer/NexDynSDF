#!/usr/bin/env python3
"""
Render model images for paper figures and export a TeX snippet.

Supported inputs:
- .nsm (custom loader)
- formats supported by pyvista.read (e.g. .obj, .vtp, .ply, .stl)
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYVISTA_USE_IPYVTK", "false")

try:
    import pyvista as pv
except Exception as e:  # pragma: no cover
    raise RuntimeError("pyvista is required for model rendering") from e

pv.OFF_SCREEN = True


def log(msg: str) -> None:
    print(f"[render] {msg}")


def parse_model_list(models: Sequence[str], models_file: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for m in models:
        out.append(Path(m))
    if models_file:
        mf = Path(models_file)
        if not mf.exists():
            raise FileNotFoundError(f"models file not found: {mf}")
        for line in mf.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(Path(s))
    if not out:
        raise ValueError("no models provided")
    uniq: List[Path] = []
    seen = set()
    for p in out:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def load_nsm_polydata(path: Path) -> pv.PolyData:
    with path.open("rb") as f:
        head = f.read(64)
        if len(head) != 64:
            raise RuntimeError(f"invalid NSM header: {path}")
        if head[0:4] != b"NSM\x00":
            raise RuntimeError(f"invalid NSM magic: {path}")
        ver = struct.unpack("<I", head[4:8])[0]
        if ver != 1:
            raise RuntimeError(f"unsupported NSM version {ver}: {path}")
        n_vertices = struct.unpack("<I", head[8:12])[0]
        n_triangles = struct.unpack("<I", head[12:16])[0]

        vertices = np.fromfile(f, dtype=np.float64, count=n_vertices * 3)
        if vertices.size != n_vertices * 3:
            raise RuntimeError(f"invalid NSM vertices size: {path}")
        vertices = vertices.reshape((n_vertices, 3))

        triangles = np.fromfile(f, dtype=np.uint32, count=n_triangles * 3)
        if triangles.size != n_triangles * 3:
            raise RuntimeError(f"invalid NSM triangles size: {path}")
        triangles = triangles.reshape((n_triangles, 3))

    faces = np.empty((n_triangles, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles.astype(np.int64)
    return pv.PolyData(vertices, faces.ravel())


def load_mesh(path: Path) -> pv.PolyData:
    suffix = path.suffix.lower()
    if suffix == ".nsm":
        mesh = load_nsm_polydata(path)
    else:
        data = pv.read(str(path))
        if isinstance(data, pv.PolyData):
            mesh = data
        else:
            mesh = data.extract_surface().triangulate()
    if mesh.n_cells == 0 or mesh.n_points == 0:
        raise RuntimeError(f"empty mesh: {path}")
    if "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False)
    return mesh


def rel_tex_path(tex_file: Path, image_path: Path) -> str:
    rel = os.path.relpath(str(image_path), start=str(tex_file.parent))
    return rel.replace("\\", "/")


def render_one(
    mesh: pv.PolyData,
    out_png: Path,
    width: int,
    height: int,
    color: str,
    edge_color: str,
    show_edges: bool,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    p = pv.Plotter(window_size=[width, height], off_screen=True)
    p.set_background("white")
    p.add_mesh(
        mesh,
        color=color,
        show_edges=show_edges,
        edge_color=edge_color,
        smooth_shading=True,
        specular=0.35,
        specular_power=24,
    )
    p.enable_anti_aliasing("ssaa")
    p.camera_position = "iso"
    p.reset_camera()
    p.camera.zoom(1.15)
    p.screenshot(str(out_png), transparent_background=False)
    p.close()


def make_gallery(
    pairs: Sequence[Tuple[str, Path]],
    out_png: Path,
    cols: int,
    dpi: int = 160,
) -> None:
    if plt is None:
        log("matplotlib not available, skip gallery")
        return
    if not pairs:
        return
    cols = max(1, int(cols))
    rows = int(math.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows), dpi=dpi)
    axes_arr = np.asarray(axes).reshape(-1)

    for i, (name, path) in enumerate(pairs):
        ax = axes_arr[i]
        img = plt.imread(str(path))
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    for j in range(len(pairs), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def write_tex_snippet(
    pairs: Sequence[Tuple[str, Path]],
    tex_out: Path,
    cols: int,
    caption: str,
    label: str,
) -> None:
    cols = max(1, int(cols))
    lines: List[str] = []
    lines.append("\\begin{figure}[t]")
    lines.append("\\centering")

    idx = 0
    while idx < len(pairs):
        chunk = pairs[idx : idx + cols]
        width = 0.96 / max(1, len(chunk))
        img_row: List[str] = []
        name_row: List[str] = []
        for name, p in chunk:
            include_path = rel_tex_path(tex_out, p)
            img_row.append(f"\\includegraphics[width={width:.3f}\\linewidth]{{{include_path}}}")
            name_row.append(name.replace("_", "\\_"))
        lines.append(" \\hfill ".join(img_row) + " \\\\")
        lines.append(" \\hfill ".join(name_row) + " \\\\")
        lines.append("\\vspace{2mm}")
        idx += cols

    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{figure}")

    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render model images and generate TeX snippet")
    parser.add_argument("--models", nargs="*", default=[], help="model file paths")
    parser.add_argument("--models_file", default=None, help="text file with one model path per line")
    parser.add_argument("--out_dir", default="output/paper1_rmse/model_images", help="image output dir")
    parser.add_argument("--tex_out", default=None, help="output TeX snippet path")
    parser.add_argument("--gallery_out", default=None, help="output gallery png path")
    parser.add_argument("--cols", type=int, default=3, help="columns for gallery and TeX layout")
    parser.add_argument("--width", type=int, default=1280, help="single image width")
    parser.add_argument("--height", type=int, default=960, help="single image height")
    parser.add_argument("--show_edges", action="store_true", help="render mesh edges")
    parser.add_argument("--color", default="#87B3D6", help="mesh color")
    parser.add_argument("--edge_color", default="#1F2D3D", help="edge color")
    parser.add_argument("--caption", default="Benchmark models used in experiments.")
    parser.add_argument("--label", default="fig:benchmark_models_auto")
    parser.add_argument("--force", action="store_true", help="rerender existing png files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    tex_out = Path(args.tex_out) if args.tex_out else (out_dir / "fig_models_auto.tex")
    if not tex_out.is_absolute():
        tex_out = (repo_root / tex_out).resolve()

    gallery_out = (
        Path(args.gallery_out) if args.gallery_out else (out_dir / "models_gallery.png")
    )
    if not gallery_out.is_absolute():
        gallery_out = (repo_root / gallery_out).resolve()

    try:
        models = parse_model_list(args.models, args.models_file)
    except Exception as e:
        log(f"input error: {e}")
        return 2

    pairs: List[Tuple[str, Path]] = []
    for m in models:
        model_abs = m if m.is_absolute() else (repo_root / m)
        model_abs = model_abs.resolve()
        if not model_abs.exists():
            log(f"skip missing model: {model_abs}")
            continue
        out_png = out_dir / f"{model_abs.stem}.png"
        try:
            if out_png.exists() and not args.force:
                log(f"reuse image: {out_png}")
            else:
                mesh = load_mesh(model_abs)
                render_one(
                    mesh=mesh,
                    out_png=out_png,
                    width=args.width,
                    height=args.height,
                    color=args.color,
                    edge_color=args.edge_color,
                    show_edges=args.show_edges,
                )
                log(f"rendered: {out_png}")
            pairs.append((model_abs.stem, out_png))
        except Exception as e:
            log(f"failed {model_abs}: {e}")

    if not pairs:
        log("no images generated")
        return 1

    make_gallery(pairs, gallery_out, args.cols)
    write_tex_snippet(
        pairs=pairs,
        tex_out=tex_out,
        cols=args.cols,
        caption=args.caption,
        label=args.label,
    )
    log(f"models rendered: {len(pairs)}")
    log(f"gallery: {gallery_out}")
    log(f"tex snippet: {tex_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
