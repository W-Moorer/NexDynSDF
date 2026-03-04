#!/usr/bin/env python3
"""
Paper-1 RMSE/Linf experiment pipeline.

One command to:
1) build SDFs for selected methods
2) sample to RAW grids
3) compute RMSE / L_inf against exact reference
4) export CSV summaries
5) generate LaTeX table (paper template style)
6) generate plots

Default methods are intentionally lightweight:
- Planar  -> Octree SDF
- Ours    -> Hybrid SDF (NSM only)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shlex
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from analytic_bench_eval import (
        evaluate_analytic_points_against_raw,
        load_analytic_metadata,
    )
except Exception:  # pragma: no cover
    evaluate_analytic_points_against_raw = None
    load_analytic_metadata = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PAPER_TABLE_ORDER = ["Planar", "Subdiv-Planar", "Nagata", "Tess-Patch", "Ours"]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    paper_name: str
    sdf_format: str
    requires_nsm: bool = False
    is_reference: bool = False
    preprocess: str = "none"  # none | subdiv_planar | tess_patch
    hybrid_disable_enhancement: bool = False


METHODS: Dict[str, MethodSpec] = {
    "planar": MethodSpec("planar", "Planar", "octree"),
    "subdiv_planar": MethodSpec(
        "subdiv_planar", "Subdiv-Planar", "octree", preprocess="subdiv_planar"
    ),
    "nagata": MethodSpec(
        "nagata",
        "Nagata",
        "hybrid",
        requires_nsm=True,
        hybrid_disable_enhancement=True,
    ),
    "tess_patch": MethodSpec(
        "tess_patch",
        "Tess-Patch",
        "octree",
        requires_nsm=True,
        preprocess="tess_patch",
    ),
    "ours": MethodSpec("ours", "Ours", "hybrid", requires_nsm=True),
    "exact_ref": MethodSpec("exact_ref", "ExactRef", "exact_octree", is_reference=True),
}


def log(msg: str) -> None:
    print(f"[paper1] {msg}")


def stable_seed(base_seed: int, *parts: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False) % 2147483647


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
    # keep order, remove duplicates
    uniq: List[Path] = []
    seen = set()
    for p in out:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def merge_model_lists(*lists: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for seq in lists:
        for p in seq:
            k = str(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return out


def resolve_path(repo_root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def find_executable(explicit: Optional[str], repo_root: Path, candidates: Sequence[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"executable not found: {p}")

    search_roots = [
        repo_root / "build",
        repo_root / "build" / "Release",
        repo_root / "build" / "Debug",
    ]
    for root in search_roots:
        for name in candidates:
            p = root / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"cannot find executable. tried: {', '.join(candidates)} under {search_roots}"
    )


def run_cmd(cmd: Sequence[str], cwd: Path, verbose: bool = False) -> None:
    cmd_list = [str(c) for c in cmd]
    show = " ".join(shlex.quote(c) for c in cmd_list)
    log(f"RUN: {show}")
    proc = subprocess.run(
        cmd_list,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if verbose or proc.returncode != 0:
        print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd_list)}")


def load_nsm_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = f.read(64)
        if len(header) != 64:
            raise RuntimeError(f"invalid NSM header: {path}")
        magic = header[0:4]
        version = struct.unpack("<I", header[4:8])[0]
        n_vertices = struct.unpack("<I", header[8:12])[0]
        n_triangles = struct.unpack("<I", header[12:16])[0]
        if magic != b"NSM\x00":
            raise RuntimeError(f"invalid NSM magic: {path}")
        if version != 1:
            raise RuntimeError(f"unsupported NSM version={version}: {path}")

        vertices = np.fromfile(f, dtype=np.float64, count=n_vertices * 3)
        if vertices.size != n_vertices * 3:
            raise RuntimeError(f"incomplete NSM vertices: {path}")
        faces = np.fromfile(f, dtype=np.uint32, count=n_triangles * 3)
        if faces.size != n_triangles * 3:
            raise RuntimeError(f"incomplete NSM faces: {path}")
    return vertices.reshape(n_vertices, 3), faces.reshape(n_triangles, 3)


def load_obj_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = []
                for tok in parts[1:4]:
                    head = tok.split("/")[0]
                    idx.append(int(head) - 1)
                faces.append(idx)
    if not vertices or not faces:
        raise RuntimeError(f"invalid OBJ mesh: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def subdivide_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    levels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    levels = max(0, int(levels))
    out_v = vertices.astype(np.float64, copy=True)
    out_f = faces.astype(np.uint32, copy=True)
    for _ in range(levels):
        edge_mid: Dict[Tuple[int, int], int] = {}
        new_vertices = out_v.tolist()
        new_faces: List[List[int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in edge_mid:
                return edge_mid[key]
            p = 0.5 * (out_v[a] + out_v[b])
            idx = len(new_vertices)
            new_vertices.append([float(p[0]), float(p[1]), float(p[2])])
            edge_mid[key] = idx
            return idx

        for tri in out_f:
            a = int(tri[0])
            b = int(tri[1])
            c = int(tri[2])
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([ab, b, bc])
            new_faces.append([ca, bc, c])
            new_faces.append([ab, bc, ca])

        out_v = np.asarray(new_vertices, dtype=np.float64)
        out_f = np.asarray(new_faces, dtype=np.uint32)
    return out_v, out_f


def write_obj_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {float(v[0]):.16g} {float(v[1]):.16g} {float(v[2]):.16g}\n")
        for tri in faces:
            a = int(tri[0]) + 1
            b = int(tri[1]) + 1
            c = int(tri[2]) + 1
            f.write(f"f {a} {b} {c}\n")


def prepare_subdiv_planar_mesh(
    input_model: Path,
    output_obj: Path,
    level: int,
    force: bool,
) -> Path:
    if output_obj.exists() and not force:
        return output_obj
    ext = input_model.suffix.lower()
    if ext == ".nsm":
        v, f = load_nsm_vertices_faces(input_model)
    elif ext == ".obj":
        v, f = load_obj_vertices_faces(input_model)
    else:
        raise RuntimeError(f"subdiv_planar supports .nsm/.obj only: {input_model}")
    sv, sf = subdivide_triangles(v, f, level)
    write_obj_mesh(output_obj, sv, sf)
    return output_obj


def prepare_tess_patch_mesh(
    nagata_exporter: Path,
    repo_root: Path,
    input_model: Path,
    output_dir: Path,
    level: int,
    tolerance: float,
    force: bool,
    verbose: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_obj = output_dir / f"{input_model.stem}_enhanced_L{int(level)}.obj"
    if out_obj.exists() and not force:
        return out_obj
    cmd = [
        str(nagata_exporter),
        str(input_model),
        str(output_dir),
        str(int(level)),
        str(float(tolerance)),
    ]
    run_cmd(cmd, cwd=repo_root, verbose=verbose)
    if not out_obj.exists():
        raise RuntimeError(f"NagataExporter did not generate expected file: {out_obj}")
    return out_obj


def load_raw_grid(path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        res_arr = np.fromfile(f, dtype=np.int32, count=1)
        if res_arr.size != 1:
            raise RuntimeError(f"invalid raw header: {path}")
        res = int(res_arr[0])
        bbox = np.fromfile(f, dtype=np.float32, count=6)
        if bbox.size != 6:
            raise RuntimeError(f"invalid raw bbox: {path}")
        count = res * res * res
        vals = np.fromfile(f, dtype=np.float32, count=count)
        if vals.size != count:
            raise RuntimeError(
                f"invalid raw data size: {path}, expect {count}, got {vals.size}"
            )
    bbox_min = bbox[:3].astype(np.float64)
    bbox_max = bbox[3:].astype(np.float64)
    grid = vals.reshape((res, res, res), order="F")
    return res, bbox_min, bbox_max, grid


def grids_compatible(
    res_a: int,
    bmin_a: np.ndarray,
    bmax_a: np.ndarray,
    res_b: int,
    bmin_b: np.ndarray,
    bmax_b: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    if res_a != res_b:
        return False
    return bool(
        np.max(np.abs(bmin_a - bmin_b)) <= tol and np.max(np.abs(bmax_a - bmax_b)) <= tol
    )


def compute_rmse_linf(
    pred: np.ndarray, ref: np.ndarray, narrowband: Optional[float]
) -> Tuple[float, float, int]:
    if pred.shape != ref.shape:
        raise ValueError("grid shape mismatch")
    if narrowband is not None and narrowband > 0.0:
        mask = np.abs(ref) <= narrowband
        if not np.any(mask):
            raise ValueError("narrowband mask is empty")
        diff = pred[mask] - ref[mask]
    else:
        diff = pred - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))
    return rmse, linf, int(diff.size)


def format_metric(x: Optional[float]) -> str:
    if x is None:
        return "--"
    if not math.isfinite(x):
        return "--"
    return f"{x:.6e}"


def format_mean_std(mean_v: Optional[float], std_v: Optional[float]) -> str:
    if mean_v is None:
        return "--"
    if not math.isfinite(mean_v):
        return "--"
    if std_v is None or not math.isfinite(std_v):
        return f"{mean_v:.6e}"
    return f"{mean_v:.6e} $\\pm$ {std_v:.6e}"


def tex_rel_path(tex_file: Path, target: Path) -> str:
    rel = os.path.relpath(str(target), start=str(tex_file.parent))
    return rel.replace("\\", "/")


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: Iterable[dict]) -> List[dict]:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    grouped_norm_mean: Dict[str, List[float]] = {}
    grouped_norm_rmse: Dict[str, List[float]] = {}
    grouped_norm_p95: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("is_reference") == "1":
            continue
        method = str(r["paper_method"])
        grouped.setdefault(method, []).append((float(r["rmse"]), float(r["linf"])))
        nmean_s = str(r.get("normal_mean_deg", "")).strip()
        nrmse_s = str(r.get("normal_rmse_deg", "")).strip()
        np95_s = str(r.get("normal_p95_deg", "")).strip()
        if nmean_s and nrmse_s and np95_s:
            try:
                nmean = float(nmean_s)
                nrmse = float(nrmse_s)
                np95 = float(np95_s)
                if math.isfinite(nmean) and math.isfinite(nrmse) and math.isfinite(np95):
                    grouped_norm_mean.setdefault(method, []).append(nmean)
                    grouped_norm_rmse.setdefault(method, []).append(nrmse)
                    grouped_norm_p95.setdefault(method, []).append(np95)
            except Exception:
                pass

    out: List[dict] = []
    for method, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        norm_mean_vals = grouped_norm_mean.get(method, [])
        norm_rmse_vals = grouped_norm_rmse.get(method, [])
        norm_p95_vals = grouped_norm_p95.get(method, [])
        out.append(
            {
                "paper_method": method,
                "rmse_mean": float(np.mean(arr[:, 0])),
                "rmse_std": float(np.std(arr[:, 0])),
                "linf_mean": float(np.mean(arr[:, 1])),
                "linf_std": float(np.std(arr[:, 1])),
                "models_count": int(arr.shape[0]),
                "normal_models_count": int(len(norm_mean_vals)),
                "normal_mean_deg": float(np.mean(norm_mean_vals)) if norm_mean_vals else float("nan"),
                "normal_rmse_deg": float(np.mean(norm_rmse_vals)) if norm_rmse_vals else float("nan"),
                "normal_p95_deg": float(np.mean(norm_p95_vals)) if norm_p95_vals else float("nan"),
            }
        )
    out.sort(key=lambda x: PAPER_TABLE_ORDER.index(x["paper_method"]) if x["paper_method"] in PAPER_TABLE_ORDER else 999)
    return out


def aggregate_seed_summary(rows: Iterable[dict]) -> List[dict]:
    grouped: Dict[Tuple[int, str], List[Tuple[float, float]]] = {}
    grouped_norm_mean: Dict[Tuple[int, str], List[float]] = {}
    grouped_norm_rmse: Dict[Tuple[int, str], List[float]] = {}
    grouped_norm_p95: Dict[Tuple[int, str], List[float]] = {}

    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("is_reference") == "1":
            continue
        seed_s = str(r.get("analytic_seed", "")).strip()
        if not seed_s:
            continue
        try:
            seed = int(seed_s)
        except Exception:
            continue
        method = str(r["paper_method"])
        key = (seed, method)
        grouped.setdefault(key, []).append((float(r["rmse"]), float(r["linf"])))

        nmean_s = str(r.get("normal_mean_deg", "")).strip()
        nrmse_s = str(r.get("normal_rmse_deg", "")).strip()
        np95_s = str(r.get("normal_p95_deg", "")).strip()
        if nmean_s and nrmse_s and np95_s:
            try:
                nmean = float(nmean_s)
                nrmse = float(nrmse_s)
                np95 = float(np95_s)
                if math.isfinite(nmean) and math.isfinite(nrmse) and math.isfinite(np95):
                    grouped_norm_mean.setdefault(key, []).append(nmean)
                    grouped_norm_rmse.setdefault(key, []).append(nrmse)
                    grouped_norm_p95.setdefault(key, []).append(np95)
            except Exception:
                pass

    out: List[dict] = []
    for key, vals in grouped.items():
        seed, method = key
        arr = np.asarray(vals, dtype=np.float64)
        norm_mean_vals = grouped_norm_mean.get(key, [])
        norm_rmse_vals = grouped_norm_rmse.get(key, [])
        norm_p95_vals = grouped_norm_p95.get(key, [])
        out.append(
            {
                "analytic_seed": int(seed),
                "paper_method": method,
                "rmse_mean": float(np.mean(arr[:, 0])),
                "linf_mean": float(np.mean(arr[:, 1])),
                "models_count": int(arr.shape[0]),
                "normal_models_count": int(len(norm_mean_vals)),
                "normal_mean_deg": float(np.mean(norm_mean_vals)) if norm_mean_vals else float("nan"),
                "normal_rmse_deg": float(np.mean(norm_rmse_vals)) if norm_rmse_vals else float("nan"),
                "normal_p95_deg": float(np.mean(norm_p95_vals)) if norm_p95_vals else float("nan"),
            }
        )
    out.sort(
        key=lambda x: (
            int(x["analytic_seed"]),
            PAPER_TABLE_ORDER.index(x["paper_method"])
            if x["paper_method"] in PAPER_TABLE_ORDER
            else 999,
        )
    )
    return out


def aggregate_multiseed_summary(seed_rows: Iterable[dict]) -> List[dict]:
    rmse_map: Dict[str, List[float]] = {}
    linf_map: Dict[str, List[float]] = {}
    nmean_map: Dict[str, List[float]] = {}
    nrmse_map: Dict[str, List[float]] = {}
    np95_map: Dict[str, List[float]] = {}
    seeds_map: Dict[str, set] = {}

    for r in seed_rows:
        method = str(r["paper_method"])
        seed = int(r["analytic_seed"])
        seeds_map.setdefault(method, set()).add(seed)
        try:
            rmse_v = float(r["rmse_mean"])
            linf_v = float(r["linf_mean"])
            if math.isfinite(rmse_v):
                rmse_map.setdefault(method, []).append(rmse_v)
            if math.isfinite(linf_v):
                linf_map.setdefault(method, []).append(linf_v)
        except Exception:
            pass
        for key, dst in [
            ("normal_mean_deg", nmean_map),
            ("normal_rmse_deg", nrmse_map),
            ("normal_p95_deg", np95_map),
        ]:
            s = str(r.get(key, "")).strip()
            if not s:
                continue
            try:
                v = float(s)
            except Exception:
                continue
            if math.isfinite(v):
                dst.setdefault(method, []).append(v)

    all_methods = set(seeds_map.keys()) | set(rmse_map.keys()) | set(linf_map.keys())
    out: List[dict] = []
    for method in all_methods:
        rmse_vals = np.asarray(rmse_map.get(method, []), dtype=np.float64)
        linf_vals = np.asarray(linf_map.get(method, []), dtype=np.float64)
        nmean_vals = np.asarray(nmean_map.get(method, []), dtype=np.float64)
        nrmse_vals = np.asarray(nrmse_map.get(method, []), dtype=np.float64)
        np95_vals = np.asarray(np95_map.get(method, []), dtype=np.float64)
        out.append(
            {
                "paper_method": method,
                "seeds_count": int(len(seeds_map.get(method, set()))),
                "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals.size > 0 else float("nan"),
                "rmse_std": float(np.std(rmse_vals)) if rmse_vals.size > 0 else float("nan"),
                "linf_mean": float(np.mean(linf_vals)) if linf_vals.size > 0 else float("nan"),
                "linf_std": float(np.std(linf_vals)) if linf_vals.size > 0 else float("nan"),
                "normal_mean_deg_mean": float(np.mean(nmean_vals)) if nmean_vals.size > 0 else float("nan"),
                "normal_mean_deg_std": float(np.std(nmean_vals)) if nmean_vals.size > 0 else float("nan"),
                "normal_rmse_deg_mean": float(np.mean(nrmse_vals)) if nrmse_vals.size > 0 else float("nan"),
                "normal_rmse_deg_std": float(np.std(nrmse_vals)) if nrmse_vals.size > 0 else float("nan"),
                "normal_p95_deg_mean": float(np.mean(np95_vals)) if np95_vals.size > 0 else float("nan"),
                "normal_p95_deg_std": float(np.std(np95_vals)) if np95_vals.size > 0 else float("nan"),
            }
        )
    out.sort(key=lambda x: PAPER_TABLE_ORDER.index(x["paper_method"]) if x["paper_method"] in PAPER_TABLE_ORDER else 999)
    return out


def write_latex_table(summary_rows: List[dict], out_path: Path) -> None:
    by_method = {r["paper_method"]: r for r in summary_rows}
    has_normal = any(
        math.isfinite(float(r.get("normal_mean_deg", float("nan"))))
        for r in summary_rows
    )
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    if has_normal:
        lines.append(
            "\\caption{Surface-point SDF-zero error and normal-angle comparison (auto-generated)}"
        )
    else:
        lines.append("\\caption{SDF error comparison (RMSE and $L_\\infty$, auto-generated)}")
    lines.append("\\label{tab:sdf_rmse_linf_auto}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{lccccc}" if has_normal else "\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    if has_normal:
        lines.append(
            "Method & Surface-$|SDF|$ RMSE$\\downarrow$ & Surface-$|SDF|$ $L_\\infty\\downarrow$ & Normal Mean$\\downarrow$ & Normal RMSE$\\downarrow$ & Normal P95$\\downarrow$ \\\\"
        )
    else:
        lines.append("Method & Surface-$|SDF|$ RMSE$\\downarrow$ & Surface-$|SDF|$ $L_\\infty\\downarrow$ \\\\")
    lines.append("\\midrule")
    for m in PAPER_TABLE_ORDER:
        row = by_method.get(m)
        rmse = format_metric(row["rmse_mean"]) if row else "--"
        linf = format_metric(row["linf_mean"]) if row else "--"
        if has_normal:
            nmean = format_metric(row["normal_mean_deg"]) if row else "--"
            nrmse = format_metric(row["normal_rmse_deg"]) if row else "--"
            np95 = format_metric(row["normal_p95_deg"]) if row else "--"
            lines.append(f"{m} & {rmse} & {linf} & {nmean} & {nrmse} & {np95} \\\\")
        else:
            lines.append(f"{m} & {rmse} & {linf} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_multiseed_table(multiseed_rows: List[dict], out_path: Path) -> None:
    by_method = {r["paper_method"]: r for r in multiseed_rows}
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Multi-seed stability on analytic surface points (mean$\\pm$std over seeds, auto-generated)}"
    )
    lines.append("\\label{tab:sdf_rmse_linf_multiseed_auto}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "Method & Surface-$|SDF|$ RMSE$\\downarrow$ (mean$\\pm$std) & Surface-$|SDF|$ $L_\\infty\\downarrow$ (mean$\\pm$std) & Normal RMSE$\\downarrow$ (mean$\\pm$std) & Normal P95$\\downarrow$ (mean$\\pm$std) & $N_{seed}$ \\\\"
    )
    lines.append("\\midrule")
    for m in PAPER_TABLE_ORDER:
        row = by_method.get(m)
        if row is None:
            lines.append(f"{m} & -- & -- & -- & -- & -- \\\\")
            continue
        rmse = format_mean_std(float(row["rmse_mean"]), float(row["rmse_std"]))
        linf = format_mean_std(float(row["linf_mean"]), float(row["linf_std"]))
        nrmse = format_mean_std(
            float(row["normal_rmse_deg_mean"]), float(row["normal_rmse_deg_std"])
        )
        np95 = format_mean_std(float(row["normal_p95_deg_mean"]), float(row["normal_p95_deg_std"]))
        nseed = str(int(row["seeds_count"]))
        lines.append(f"{m} & {rmse} & {linf} & {nrmse} & {np95} & {nseed} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_normal_plot_tex(tex_path: Path, image_path: Path) -> None:
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.92\\linewidth]{{{tex_rel_path(tex_path, image_path)}}}",
        "\\caption{Surface-point normal-angle RMSE and P95 comparison (auto-generated).}",
        "\\label{fig:normal_rmse_p95_auto}",
        "\\end{figure}",
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary_rows: List[dict], out_dir: Path, log_scale: bool = True) -> None:
    if plt is None:
        log("matplotlib not available, skip plotting")
        return
    if not summary_rows:
        log("no summary rows, skip plotting")
        return

    methods = [r["paper_method"] for r in summary_rows]
    rmse = np.asarray([r["rmse_mean"] for r in summary_rows], dtype=np.float64)
    linf = np.asarray([r["linf_mean"] for r in summary_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.4 * len(methods)), 4.2), dpi=150)
    x = np.arange(len(methods))

    axes[0].bar(x, rmse, color="#4C78A8")
    axes[0].set_title("RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("Error")
    if log_scale:
        axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, linf, color="#F58518")
    axes[1].set_title("L_inf")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    if log_scale:
        axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rmse_linf_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_model(detailed_rows: List[dict], out_dir: Path, log_scale: bool = True) -> None:
    if plt is None:
        return
    ok = [r for r in detailed_rows if r.get("status") == "ok" and r.get("is_reference") != "1"]
    if not ok:
        return

    models = sorted({r["model_name"] for r in ok})
    methods = sorted(
        {r["paper_method"] for r in ok},
        key=lambda x: PAPER_TABLE_ORDER.index(x) if x in PAPER_TABLE_ORDER else 999,
    )
    rmse_map = {(r["model_name"], r["paper_method"]): float(r["rmse"]) for r in ok}
    linf_map = {(r["model_name"], r["paper_method"]): float(r["linf"]) for r in ok}

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.8 * len(models)), 7.0), dpi=150)
    x = np.arange(len(models))
    width = 0.8 / max(1, len(methods))
    offset0 = -0.5 * (len(methods) - 1) * width

    for i, m in enumerate(methods):
        rmse_vals = [rmse_map.get((md, m), np.nan) for md in models]
        linf_vals = [linf_map.get((md, m), np.nan) for md in models]
        xi = x + offset0 + i * width
        axes[0].bar(xi, rmse_vals, width=width, label=m)
        axes[1].bar(xi, linf_vals, width=width, label=m)

    axes[0].set_title("RMSE by Model")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("RMSE")
    if log_scale:
        axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].set_title("L_inf by Model")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("L_inf")
    if log_scale:
        axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rmse_linf_per_model.png", bbox_inches="tight")
    plt.close(fig)


def plot_normal_summary(summary_rows: List[dict], out_dir: Path) -> Optional[Path]:
    if plt is None:
        return None
    valid_rows = []
    for r in summary_rows:
        try:
            nrmse = float(r.get("normal_rmse_deg", float("nan")))
            np95 = float(r.get("normal_p95_deg", float("nan")))
        except Exception:
            continue
        if math.isfinite(nrmse) and math.isfinite(np95):
            valid_rows.append(r)
    if not valid_rows:
        return None

    methods = [str(r["paper_method"]) for r in valid_rows]
    nrmse_vals = np.asarray([float(r["normal_rmse_deg"]) for r in valid_rows], dtype=np.float64)
    np95_vals = np.asarray([float(r["normal_p95_deg"]) for r in valid_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.4 * len(methods)), 4.2), dpi=150)
    x = np.arange(len(methods))

    axes[0].bar(x, nrmse_vals, color="#54A24B")
    axes[0].set_title("Normal RMSE (deg)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, np95_vals, color="#E45756")
    axes[1].set_title("Normal P95 (deg)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "normal_rmse_p95_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_sdf(
    exporter: Path,
    repo_root: Path,
    input_model: Path,
    output_sdf: Path,
    method: MethodSpec,
    depth: int,
    start_depth: int,
    termination: float,
    num_threads: int,
    force: bool,
    verbose: bool,
) -> None:
    if output_sdf.exists() and not force:
        log(f"reuse SDF: {output_sdf}")
        return
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exporter),
        str(input_model),
        str(output_sdf),
        "--depth",
        str(depth),
        "--start_depth",
        str(start_depth),
        "--sdf_format",
        method.sdf_format,
        "--termination",
        str(termination),
        "--num_threads",
        str(num_threads),
    ]
    if method.hybrid_disable_enhancement:
        cmd.append("--hybrid_disable_enhancement")
    run_cmd(cmd, cwd=repo_root, verbose=verbose)


def sample_sdf(
    sampler: Path,
    repo_root: Path,
    input_sdf: Path,
    output_raw: Path,
    grid: int,
    force: bool,
    verbose: bool,
) -> None:
    if output_raw.exists() and not force:
        log(f"reuse RAW: {output_raw}")
        return
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(sampler), str(input_sdf), str(output_raw), str(grid)]
    run_cmd(cmd, cwd=repo_root, verbose=verbose)


def run_auto_benchmark_generation(
    repo_root: Path,
    profile: str,
    models_dir: Path,
    manifest: Path,
    metadata_csv: Path,
    force: bool,
    verbose: bool,
) -> None:
    gen_script = repo_root / "demos" / "generate_benchmark_nsm_suite.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"benchmark generator not found: {gen_script}")
    cmd = [
        sys.executable,
        str(gen_script),
        "--output_dir",
        str(models_dir),
        "--profile",
        profile,
        "--manifest",
        str(manifest),
        "--metadata_csv",
        str(metadata_csv),
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd, cwd=repo_root, verbose=verbose)


def run_model_rendering(
    repo_root: Path,
    models: Sequence[Path],
    out_dir: Path,
    tex_out: Path,
    gallery_out: Path,
    cols: int,
    force: bool,
    verbose: bool,
) -> None:
    render_script = repo_root / "pytools" / "render_model_images.py"
    if not render_script.exists():
        raise FileNotFoundError(f"render script not found: {render_script}")

    cmd = [
        sys.executable,
        str(render_script),
        "--out_dir",
        str(out_dir),
        "--tex_out",
        str(tex_out),
        "--gallery_out",
        str(gallery_out),
        "--cols",
        str(cols),
        "--models",
    ]
    cmd.extend(str(m) for m in models)
    if force:
        cmd.append("--force")
    run_cmd(cmd, cwd=repo_root, verbose=verbose)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-1 RMSE/L_inf experiment pipeline")
    parser.add_argument("--models", nargs="*", default=[], help="model file paths (.nsm/.obj/.vtp)")
    parser.add_argument("--models_file", default=None, help="text file with one model path per line")
    parser.add_argument("--out_dir", default="output/paper1_rmse", help="output directory")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["planar", "subdiv_planar", "nagata", "tess_patch", "ours"],
        choices=["planar", "subdiv_planar", "nagata", "tess_patch", "ours"],
    )
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--start_depth", type=int, default=1)
    parser.add_argument("--termination", type=float, default=1e-3)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--grid", type=int, default=128, help="SdfSampler resolution")
    parser.add_argument("--narrowband", type=float, default=None, help="optional |ref|<=band filter")
    parser.add_argument("--force", action="store_true", help="rebuild all intermediate files")
    parser.add_argument("--verbose", action="store_true", help="print tool stdout")
    parser.add_argument("--no_plot", action="store_true", help="skip plotting")
    parser.add_argument("--linear_plot", action="store_true", help="use linear y-axis in plots")
    parser.add_argument("--sdf_exporter", default=None, help="explicit SdfExporter executable path")
    parser.add_argument("--sdf_sampler", default=None, help="explicit SdfSampler executable path")
    parser.add_argument("--nagata_exporter", default=None, help="explicit NagataExporter executable path")
    parser.add_argument(
        "--subdiv_planar_level",
        type=int,
        default=2,
        help="subdivision level for Subdiv-Planar baseline (each level = x4 faces)",
    )
    parser.add_argument(
        "--tess_patch_level",
        type=int,
        default=2,
        help="subdivision level for Tess-Patch baseline",
    )
    parser.add_argument(
        "--tess_patch_tolerance",
        type=float,
        default=0.1,
        help="NagataExporter tolerance for Tess-Patch generation",
    )
    parser.add_argument(
        "--analytic_point_eval",
        action="store_true",
        help="evaluate on analytic model surface samples: |SDF(surface)-0| and normal-angle metrics",
    )
    parser.add_argument(
        "--analytic_points_per_model",
        type=int,
        default=20000,
        help="number of random analytic surface samples per model for --analytic_point_eval",
    )
    parser.add_argument(
        "--analytic_seed",
        type=int,
        default=20260304,
        help="random seed for --analytic_point_eval",
    )
    parser.add_argument(
        "--analytic_seeds",
        nargs="*",
        type=int,
        default=[],
        help="optional multi-seed list for --analytic_point_eval; example: --analytic_seeds 20260304 20260305 20260306",
    )
    parser.add_argument(
        "--analytic_metadata_csv",
        default=None,
        help="benchmark metadata csv (with category/param_json); default auto-detected from benchmark models dir",
    )
    parser.add_argument(
        "--analytic_grad_step",
        type=float,
        default=0.0,
        help="world-space finite-difference step for analytic/RAW gradients (<=0 uses auto)",
    )
    parser.add_argument(
        "--auto_benchmarks",
        action="store_true",
        help="auto-generate NSM benchmark models and append to model list",
    )
    parser.add_argument(
        "--benchmark_profile",
        choices=["quick", "full", "hard"],
        default="quick",
        help="benchmark model profile for --auto_benchmarks",
    )
    parser.add_argument(
        "--benchmark_models_dir",
        default="output/benchmarks/models",
        help="output directory for generated benchmark models",
    )
    parser.add_argument(
        "--benchmark_manifest",
        default=None,
        help="manifest txt path for generated benchmark models",
    )
    parser.add_argument(
        "--benchmark_metadata_csv",
        default=None,
        help="metadata csv path for generated benchmark models",
    )
    parser.add_argument(
        "--render_model_images",
        action="store_true",
        help="render model images and export TeX snippet for insertion",
    )
    parser.add_argument(
        "--model_images_dir",
        default=None,
        help="output directory for rendered model images",
    )
    parser.add_argument(
        "--tex_models_snippet",
        default=None,
        help="TeX snippet output path for rendered model images",
    )
    parser.add_argument(
        "--models_gallery_png",
        default=None,
        help="gallery png output path for rendered model images",
    )
    parser.add_argument(
        "--models_tex_cols",
        type=int,
        default=3,
        help="column count for model image TeX snippet/gallery",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = resolve_path(repo_root, Path(args.out_dir))
    methods = [METHODS[m] for m in args.methods]
    reference = METHODS["exact_ref"]
    analytic_seeds: List[int] = []
    seen_seed = set()
    if args.analytic_seeds:
        for s in args.analytic_seeds:
            si = int(s)
            if si in seen_seed:
                continue
            seen_seed.add(si)
            analytic_seeds.append(si)
    if not analytic_seeds:
        analytic_seeds = [int(args.analytic_seed)]

    manual_models: List[Path] = []
    try:
        if args.models or args.models_file:
            manual_models = parse_model_list(args.models, args.models_file)
    except Exception as e:
        log(f"input error: {e}")
        return 2

    benchmark_models: List[Path] = []
    benchmark_manifest: Optional[Path] = None
    benchmark_metadata_csv: Optional[Path] = None
    if args.auto_benchmarks:
        benchmark_models_dir = resolve_path(repo_root, Path(args.benchmark_models_dir))
        if args.benchmark_manifest:
            benchmark_manifest = resolve_path(repo_root, Path(args.benchmark_manifest))
        else:
            benchmark_manifest = (benchmark_models_dir / "benchmark_models.txt").resolve()
        if args.benchmark_metadata_csv:
            benchmark_metadata_csv = resolve_path(repo_root, Path(args.benchmark_metadata_csv))
        else:
            benchmark_metadata_csv = (benchmark_models_dir / "benchmark_models_meta.csv").resolve()
        try:
            run_auto_benchmark_generation(
                repo_root=repo_root,
                profile=args.benchmark_profile,
                models_dir=benchmark_models_dir,
                manifest=benchmark_manifest,
                metadata_csv=benchmark_metadata_csv,
                force=args.force,
                verbose=args.verbose,
            )
            benchmark_models = parse_model_list([], str(benchmark_manifest))
        except Exception as e:
            log(f"auto benchmark generation failed: {e}")
            return 2

    models = merge_model_lists(manual_models, benchmark_models)
    if not models:
        log("input error: no models provided (use --models/--models_file or --auto_benchmarks)")
        return 2
    resolved_models = merge_model_lists([resolve_path(repo_root, m) for m in models])

    analytic_specs: Dict[str, object] = {}
    analytic_meta_path: Optional[Path] = None
    if args.analytic_point_eval:
        if load_analytic_metadata is None or evaluate_analytic_points_against_raw is None:
            log("analytic_point_eval unavailable: failed to import analytic_bench_eval")
            return 2
        if args.narrowband is not None and args.narrowband > 0.0:
            log("note: --narrowband is ignored in analytic surface-point evaluation")
        if args.analytic_metadata_csv:
            analytic_meta_path = resolve_path(repo_root, Path(args.analytic_metadata_csv))
        elif benchmark_metadata_csv is not None:
            analytic_meta_path = benchmark_metadata_csv
        elif args.auto_benchmarks:
            analytic_meta_path = resolve_path(
                repo_root, Path(args.benchmark_models_dir) / "benchmark_models_meta.csv"
            )
        if analytic_meta_path is None:
            log("analytic_point_eval requires --analytic_metadata_csv (or --auto_benchmarks metadata)")
            return 2
        try:
            analytic_specs = load_analytic_metadata(analytic_meta_path)
        except Exception as e:
            log(f"failed to load analytic metadata: {e}")
            return 2
        log(f"analytic metadata: {analytic_meta_path} ({len(analytic_specs)} models)")

    rendered_images_dir: Optional[Path] = None
    rendered_tex_snippet: Optional[Path] = None
    rendered_gallery_png: Optional[Path] = None
    if args.render_model_images:
        model_images_dir = (
            resolve_path(repo_root, Path(args.model_images_dir))
            if args.model_images_dir
            else (out_dir / "model_images").resolve()
        )
        tex_models_snippet = (
            resolve_path(repo_root, Path(args.tex_models_snippet))
            if args.tex_models_snippet
            else (out_dir / "fig_models_auto.tex").resolve()
        )
        models_gallery_png = (
            resolve_path(repo_root, Path(args.models_gallery_png))
            if args.models_gallery_png
            else (out_dir / "models_gallery.png").resolve()
        )
        rendered_images_dir = model_images_dir
        rendered_tex_snippet = tex_models_snippet
        rendered_gallery_png = models_gallery_png
        try:
            run_model_rendering(
                repo_root=repo_root,
                models=resolved_models,
                out_dir=model_images_dir,
                tex_out=tex_models_snippet,
                gallery_out=models_gallery_png,
                cols=args.models_tex_cols,
                force=args.force,
                verbose=args.verbose,
            )
        except Exception as e:
            log(f"model rendering failed: {e}")
            return 2

    try:
        sdf_exporter = find_executable(
            args.sdf_exporter,
            repo_root,
            ["SdfExporter.exe", "SdfExporter"],
        )
        sdf_sampler = find_executable(
            args.sdf_sampler,
            repo_root,
            ["SdfSampler.exe", "SdfSampler"],
        )
    except Exception as e:
        log(str(e))
        return 2
    need_nagata_exporter = any(m.preprocess == "tess_patch" for m in methods)
    nagata_exporter: Optional[Path] = None
    if need_nagata_exporter:
        try:
            nagata_exporter = find_executable(
                args.nagata_exporter,
                repo_root,
                ["NagataExporter.exe", "NagataExporter"],
            )
        except Exception as e:
            log(str(e))
            return 2

    log(f"repo_root: {repo_root}")
    log(f"SdfExporter: {sdf_exporter}")
    log(f"SdfSampler: {sdf_sampler}")
    if nagata_exporter is not None:
        log(f"NagataExporter: {nagata_exporter}")
    log(f"models: {len(resolved_models)}")
    if benchmark_manifest is not None:
        log(f"benchmark manifest: {benchmark_manifest}")
    log(f"methods: {', '.join(m.paper_name for m in methods)}")
    if args.analytic_point_eval:
        log(f"analytic seeds: {', '.join(str(s) for s in analytic_seeds)}")
    elif args.analytic_seeds:
        log("note: --analytic_seeds is ignored without --analytic_point_eval")

    detailed_rows: List[dict] = []

    for model_abs in resolved_models:
        model_name = model_abs.stem
        if not model_abs.exists():
            log(f"skip missing model: {model_abs}")
            continue

        log(f"=== model: {model_name} ===")

        model_dir = out_dir / model_name
        sdf_dir = model_dir / "sdf"
        raw_dir = model_dir / "raw"
        use_analytic_eval = bool(args.analytic_point_eval and model_name in analytic_specs)
        if args.analytic_point_eval and model_name not in analytic_specs:
            log(
                f"analytic metadata missing for {model_name}, fallback to exact_ref grid evaluation"
            )

        ref_res = 0
        ref_bmin = np.zeros((3,), dtype=np.float64)
        ref_bmax = np.zeros((3,), dtype=np.float64)
        ref_grid = np.zeros((1, 1, 1), dtype=np.float64)
        if not use_analytic_eval:
            # reference
            ref_sdf = sdf_dir / "exact_ref.bin"
            ref_raw = raw_dir / f"exact_ref_g{args.grid}.raw"

            try:
                build_sdf(
                    exporter=sdf_exporter,
                    repo_root=repo_root,
                    input_model=model_abs,
                    output_sdf=ref_sdf,
                    method=reference,
                    depth=args.depth,
                    start_depth=args.start_depth,
                    termination=args.termination,
                    num_threads=args.num_threads,
                    force=args.force,
                    verbose=args.verbose,
                )
                sample_sdf(
                    sampler=sdf_sampler,
                    repo_root=repo_root,
                    input_sdf=ref_sdf,
                    output_raw=ref_raw,
                    grid=args.grid,
                    force=args.force,
                    verbose=args.verbose,
                )
                ref_res, ref_bmin, ref_bmax, ref_grid = load_raw_grid(ref_raw)
            except Exception as e:
                log(f"reference failed for {model_name}: {e}")
                continue

        for method in methods:
            method_sdf = sdf_dir / f"{method.key}.bin"
            method_raw = raw_dir / f"{method.key}_g{args.grid}.raw"
            method_input_model = model_abs
            base_row = {
                "model_name": model_name,
                "model_path": str(model_abs),
                "method_key": method.key,
                "paper_method": method.paper_name,
                "is_reference": "0",
                "status": "ok",
                "message": "",
                "rmse": "",
                "linf": "",
                "num_samples": "",
                "eval_mode": "analytic_surface_points" if use_analytic_eval else "grid_exact_ref",
                "analytic_seed": "",
                "normal_valid": "",
                "normal_mean_deg": "",
                "normal_rmse_deg": "",
                "normal_p95_deg": "",
                "narrowband": "" if args.narrowband is None else str(args.narrowband),
                "sdf_file": str(method_sdf),
                "raw_file": str(method_raw),
            }
            if method.requires_nsm and model_abs.suffix.lower() != ".nsm":
                if use_analytic_eval:
                    for eval_seed in analytic_seeds:
                        row = dict(base_row)
                        row["analytic_seed"] = str(int(eval_seed))
                        row["status"] = "skipped"
                        row["message"] = "hybrid requires .nsm"
                        detailed_rows.append(row)
                else:
                    row = dict(base_row)
                    row["status"] = "skipped"
                    row["message"] = "hybrid requires .nsm"
                    detailed_rows.append(row)
                continue
            if method.preprocess == "subdiv_planar":
                try:
                    derived_dir = model_dir / "derived_mesh"
                    subdiv_obj = derived_dir / f"{model_name}_subdiv_planar_L{int(args.subdiv_planar_level)}.obj"
                    method_input_model = prepare_subdiv_planar_mesh(
                        input_model=model_abs,
                        output_obj=subdiv_obj,
                        level=int(args.subdiv_planar_level),
                        force=args.force,
                    )
                except Exception as e:
                    if use_analytic_eval:
                        for eval_seed in analytic_seeds:
                            row = dict(base_row)
                            row["analytic_seed"] = str(int(eval_seed))
                            row["status"] = "failed"
                            row["message"] = str(e)
                            detailed_rows.append(row)
                    else:
                        row = dict(base_row)
                        row["status"] = "failed"
                        row["message"] = str(e)
                        detailed_rows.append(row)
                    continue
            elif method.preprocess == "tess_patch":
                if model_abs.suffix.lower() != ".nsm":
                    if use_analytic_eval:
                        for eval_seed in analytic_seeds:
                            row = dict(base_row)
                            row["analytic_seed"] = str(int(eval_seed))
                            row["status"] = "skipped"
                            row["message"] = "tess_patch requires .nsm"
                            detailed_rows.append(row)
                    else:
                        row = dict(base_row)
                        row["status"] = "skipped"
                        row["message"] = "tess_patch requires .nsm"
                        detailed_rows.append(row)
                    continue
                if nagata_exporter is None:
                    if use_analytic_eval:
                        for eval_seed in analytic_seeds:
                            row = dict(base_row)
                            row["analytic_seed"] = str(int(eval_seed))
                            row["status"] = "failed"
                            row["message"] = "NagataExporter not available"
                            detailed_rows.append(row)
                    else:
                        row = dict(base_row)
                        row["status"] = "failed"
                        row["message"] = "NagataExporter not available"
                        detailed_rows.append(row)
                    continue
                try:
                    derived_dir = model_dir / "derived_mesh"
                    method_input_model = prepare_tess_patch_mesh(
                        nagata_exporter=nagata_exporter,
                        repo_root=repo_root,
                        input_model=model_abs,
                        output_dir=derived_dir,
                        level=int(args.tess_patch_level),
                        tolerance=float(args.tess_patch_tolerance),
                        force=args.force,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    if use_analytic_eval:
                        for eval_seed in analytic_seeds:
                            row = dict(base_row)
                            row["analytic_seed"] = str(int(eval_seed))
                            row["status"] = "failed"
                            row["message"] = str(e)
                            detailed_rows.append(row)
                    else:
                        row = dict(base_row)
                        row["status"] = "failed"
                        row["message"] = str(e)
                        detailed_rows.append(row)
                    continue
            base_row["model_path"] = str(method_input_model)
            try:
                build_sdf(
                    exporter=sdf_exporter,
                    repo_root=repo_root,
                    input_model=method_input_model,
                    output_sdf=method_sdf,
                    method=method,
                    depth=args.depth,
                    start_depth=args.start_depth,
                    termination=args.termination,
                    num_threads=args.num_threads,
                    force=args.force,
                    verbose=args.verbose,
                )
                sample_sdf(
                    sampler=sdf_sampler,
                    repo_root=repo_root,
                    input_sdf=method_sdf,
                    output_raw=method_raw,
                    grid=args.grid,
                    force=args.force,
                    verbose=args.verbose,
                )
                res, bmin, bmax, pred_grid = load_raw_grid(method_raw)
                if use_analytic_eval:
                    spec = analytic_specs[model_name]
                    for eval_seed in analytic_seeds:
                        row = dict(base_row)
                        row["analytic_seed"] = str(int(eval_seed))
                        analytic = evaluate_analytic_points_against_raw(
                            spec=spec,
                            res=res,
                            bbox_min=bmin,
                            bbox_max=bmax,
                            grid=pred_grid,
                            n_points=args.analytic_points_per_model,
                            seed=stable_seed(int(eval_seed), model_name, method.key),
                            grad_step_world=args.analytic_grad_step if args.analytic_grad_step > 0.0 else None,
                            narrowband=args.narrowband,
                        )
                        row["rmse"] = f"{analytic['rmse']:.12e}"
                        row["linf"] = f"{analytic['linf']:.12e}"
                        row["num_samples"] = str(int(analytic["num_samples"]))
                        row["normal_valid"] = str(int(analytic["normal_valid"]))
                        row["normal_mean_deg"] = f"{analytic['normal_mean_deg']:.12e}"
                        row["normal_rmse_deg"] = f"{analytic['normal_rmse_deg']:.12e}"
                        row["normal_p95_deg"] = f"{analytic['normal_p95_deg']:.12e}"
                        detailed_rows.append(row)
                else:
                    row = dict(base_row)
                    if not grids_compatible(res, bmin, bmax, ref_res, ref_bmin, ref_bmax):
                        raise RuntimeError("raw grids are not aligned with reference")
                    rmse, linf, n_samples = compute_rmse_linf(
                        pred_grid, ref_grid, args.narrowband
                    )
                    row["rmse"] = f"{rmse:.12e}"
                    row["linf"] = f"{linf:.12e}"
                    row["num_samples"] = str(n_samples)
                    detailed_rows.append(row)
            except Exception as e:
                if use_analytic_eval:
                    for eval_seed in analytic_seeds:
                        row = dict(base_row)
                        row["analytic_seed"] = str(int(eval_seed))
                        row["status"] = "failed"
                        row["message"] = str(e)
                        detailed_rows.append(row)
                else:
                    row = dict(base_row)
                    row["status"] = "failed"
                    row["message"] = str(e)
                    detailed_rows.append(row)

    if not detailed_rows:
        log("no result rows produced")
        return 1

    detailed_csv = out_dir / "metrics_detailed.csv"
    summary_csv = out_dir / "metrics_summary.csv"
    latex_table = out_dir / "table_sdf_rmse_linf.tex"
    seed_summary_csv = out_dir / "metrics_seed_summary.csv"
    multiseed_summary_csv = out_dir / "metrics_multiseed_summary.csv"
    multiseed_latex_table = out_dir / "table_sdf_rmse_linf_multiseed.tex"
    normal_plot_tex = out_dir / "fig_normal_rmse_p95_auto.tex"
    normal_plot_png: Optional[Path] = None

    detailed_fields = [
        "model_name",
        "model_path",
        "method_key",
        "paper_method",
        "is_reference",
        "status",
        "message",
        "rmse",
        "linf",
        "num_samples",
        "eval_mode",
        "analytic_seed",
        "normal_valid",
        "normal_mean_deg",
        "normal_rmse_deg",
        "normal_p95_deg",
        "narrowband",
        "sdf_file",
        "raw_file",
    ]
    write_csv(detailed_csv, detailed_rows, detailed_fields)

    summary_rows = aggregate_summary(detailed_rows)
    summary_fields = [
        "paper_method",
        "rmse_mean",
        "rmse_std",
        "linf_mean",
        "linf_std",
        "models_count",
        "normal_models_count",
        "normal_mean_deg",
        "normal_rmse_deg",
        "normal_p95_deg",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)
    write_latex_table(summary_rows, latex_table)

    seed_summary_rows: List[dict] = []
    multiseed_rows: List[dict] = []
    if args.analytic_point_eval:
        seed_summary_rows = aggregate_seed_summary(detailed_rows)
        seed_summary_fields = [
            "analytic_seed",
            "paper_method",
            "rmse_mean",
            "linf_mean",
            "models_count",
            "normal_models_count",
            "normal_mean_deg",
            "normal_rmse_deg",
            "normal_p95_deg",
        ]
        write_csv(seed_summary_csv, seed_summary_rows, seed_summary_fields)
        multiseed_rows = aggregate_multiseed_summary(seed_summary_rows)
        multiseed_fields = [
            "paper_method",
            "seeds_count",
            "rmse_mean",
            "rmse_std",
            "linf_mean",
            "linf_std",
            "normal_mean_deg_mean",
            "normal_mean_deg_std",
            "normal_rmse_deg_mean",
            "normal_rmse_deg_std",
            "normal_p95_deg_mean",
            "normal_p95_deg_std",
        ]
        write_csv(multiseed_summary_csv, multiseed_rows, multiseed_fields)
        write_latex_multiseed_table(multiseed_rows, multiseed_latex_table)

    if not args.no_plot:
        plot_summary(summary_rows, out_dir, log_scale=not args.linear_plot)
        plot_per_model(detailed_rows, out_dir, log_scale=not args.linear_plot)
        normal_plot_png = plot_normal_summary(summary_rows, out_dir)
        if normal_plot_png is not None:
            write_normal_plot_tex(normal_plot_tex, normal_plot_png)

    ok_rows = [r for r in detailed_rows if r["status"] == "ok"]
    fail_rows = [r for r in detailed_rows if r["status"] == "failed"]
    skip_rows = [r for r in detailed_rows if r["status"] == "skipped"]

    log(f"done. ok={len(ok_rows)}, failed={len(fail_rows)}, skipped={len(skip_rows)}")
    log(f"detailed csv: {detailed_csv}")
    log(f"summary  csv: {summary_csv}")
    log(f"latex table : {latex_table}")
    if args.analytic_point_eval:
        log(f"seed summary: {seed_summary_csv}")
        log(f"multi summary: {multiseed_summary_csv}")
        log(f"multi table : {multiseed_latex_table}")
    if not args.no_plot:
        log(f"plots       : {out_dir / 'rmse_linf_summary.png'}, {out_dir / 'rmse_linf_per_model.png'}")
        if normal_plot_png is not None:
            log(f"normal plot : {normal_plot_png}")
            log(f"normal tex  : {normal_plot_tex}")
    if rendered_images_dir is not None:
        log(f"model images: {rendered_images_dir}")
    if rendered_gallery_png is not None:
        log(f"gallery png : {rendered_gallery_png}")
    if rendered_tex_snippet is not None:
        log(f"tex snippet : {rendered_tex_snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
