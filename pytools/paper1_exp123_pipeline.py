#!/usr/bin/env python3
"""
Paper-1 experiment bundle for 1+2+3:
1) Convergence sweep (depth/grid)
2) Hard-case parameter scan
3) Ablation (termination threshold)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


BASE_METHOD_ORDER = ["Planar", "Ours"]


def log(msg: str) -> None:
    print(f"[exp123] {msg}")


def run_cmd(cmd: Sequence[str], cwd: Path, verbose: bool = False) -> None:
    cmd_list = [str(c) for c in cmd]
    shown = " ".join(shlex.quote(c) for c in cmd_list)
    log(f"RUN: {shown}")
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


def parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    tokens = re.split(r"[,\s;]+", text.strip()) if text.strip() else []
    out: List[int] = []
    seen = set()
    for tok in tokens:
        if not tok:
            continue
        v = int(tok)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if out:
        return out
    d = []
    seen.clear()
    for x in default:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        d.append(xi)
    return d


def resolve_path(repo_root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return x


def format_mean_std(mean_v: float, std_v: float) -> str:
    if not math.isfinite(mean_v):
        return "--"
    if not math.isfinite(std_v):
        return f"{mean_v:.6e}"
    return f"{mean_v:.6e} $\\pm$ {std_v:.6e}"


def tex_rel_path(tex_file: Path, target: Path) -> str:
    rel = os.path.relpath(str(target), start=str(tex_file.parent))
    return rel.replace("\\", "/")


def write_figure_tex(tex_path: Path, image_path: Path, caption: str, label: str) -> None:
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.96\\linewidth]{{{tex_rel_path(tex_path, image_path)}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{figure}",
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_generator(
    repo_root: Path,
    output_dir: Path,
    profile: str,
    manifest: Path,
    metadata_csv: Path,
    force: bool,
    verbose: bool,
) -> None:
    script = repo_root / "demos" / "generate_benchmark_nsm_suite.py"
    cmd = [
        sys.executable,
        str(script),
        "--output_dir",
        str(output_dir),
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


def run_rmse_pipeline(
    repo_root: Path,
    out_dir: Path,
    models_file: Path,
    metadata_csv: Path,
    depth: int,
    grid: int,
    termination: float,
    start_depth: int,
    num_threads: int,
    points_per_model: int,
    seeds: Sequence[int],
    methods: Sequence[str],
    force: bool,
    no_plot: bool,
    verbose: bool,
) -> None:
    if not methods:
        raise ValueError("methods is empty")
    if not seeds:
        raise ValueError("seeds is empty")

    script = repo_root / "pytools" / "paper1_rmse_pipeline.py"
    cmd = [
        sys.executable,
        str(script),
        "--models_file",
        str(models_file),
        "--out_dir",
        str(out_dir),
        "--methods",
    ]
    cmd.extend(methods)
    cmd.extend(
        [
            "--depth",
            str(int(depth)),
            "--start_depth",
            str(int(start_depth)),
            "--termination",
            str(float(termination)),
            "--grid",
            str(int(grid)),
            "--num_threads",
            str(int(num_threads)),
            "--analytic_point_eval",
            "--analytic_points_per_model",
            str(int(points_per_model)),
            "--analytic_seed",
            str(int(seeds[0])),
            "--analytic_metadata_csv",
            str(metadata_csv),
        ]
    )
    if len(seeds) > 1:
        cmd.append("--analytic_seeds")
        cmd.extend(str(int(s)) for s in seeds)
    if no_plot:
        cmd.append("--no_plot")
    if force:
        cmd.append("--force")
    run_cmd(cmd, cwd=repo_root, verbose=verbose)


def find_method_row(rows: Iterable[dict], paper_method: str) -> Optional[dict]:
    for r in rows:
        if str(r.get("paper_method", "")).strip() == paper_method:
            return r
    return None


def write_convergence_table(rows: List[dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Convergence sweep across octree depth/grid (mean$\\pm$std over seeds).}")
    lines.append("\\label{tab:exp123_convergence}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{cclcccc}")
    lines.append("\\toprule")
    lines.append(
        "Depth & Grid & Method & Surface-$|SDF|$ RMSE$\\downarrow$ & Surface-$|SDF|$ $L_\\infty\\downarrow$ & Normal RMSE$\\downarrow$ & Normal P95$\\downarrow$ \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        depth = int(r["depth"])
        grid = int(r["grid"])
        m = str(r["paper_method"])
        rmse = format_mean_std(float(r["rmse_mean"]), float(r["rmse_std"]))
        linf = format_mean_std(float(r["linf_mean"]), float(r["linf_std"]))
        nrmse = format_mean_std(float(r["normal_rmse_deg_mean"]), float(r["normal_rmse_deg_std"]))
        np95 = format_mean_std(float(r["normal_p95_deg_mean"]), float(r["normal_p95_deg_std"]))
        lines.append(f"{depth} & {grid} & {m} & {rmse} & {linf} & {nrmse} & {np95} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_convergence(rows: List[dict], out_png: Path) -> Optional[Path]:
    if plt is None:
        return None
    if not rows:
        return None

    depths = sorted({int(r["depth"]) for r in rows})
    methods = [m for m in BASE_METHOD_ORDER if any(str(r["paper_method"]) == m for r in rows)]
    grids = sorted({int(r["grid"]) for r in rows})
    if not methods or not grids or not depths:
        return None

    def pick(depth: int, grid: int, method: str, key: str) -> float:
        for rr in rows:
            if int(rr["depth"]) == depth and int(rr["grid"]) == grid and str(rr["paper_method"]) == method:
                return float(rr[key])
        return float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    for depth in depths:
        for method in methods:
            label = f"{method} d={depth}"
            y_rmse = [pick(depth, g, method, "rmse_mean") for g in grids]
            y_nrmse = [pick(depth, g, method, "normal_rmse_deg_mean") for g in grids]
            axes[0].plot(grids, y_rmse, marker="o", label=label)
            axes[1].plot(grids, y_nrmse, marker="o", label=label)

    axes[0].set_title("Surface-|SDF| RMSE vs Grid")
    axes[0].set_xlabel("Grid")
    axes[0].set_ylabel("RMSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Normal RMSE (deg) vs Grid")
    axes[1].set_xlabel("Grid")
    axes[1].set_ylabel("Normal RMSE (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_convergence_experiment(
    repo_root: Path,
    out_dir: Path,
    models_file: Path,
    metadata_csv: Path,
    depths: Sequence[int],
    grids: Sequence[int],
    seeds: Sequence[int],
    start_depth: int,
    termination: float,
    num_threads: int,
    points_per_model: int,
    force: bool,
    verbose: bool,
) -> List[dict]:
    exp_dir = out_dir / "exp1_convergence"
    rows: List[dict] = []
    for depth in depths:
        for grid in grids:
            run_out = exp_dir / f"d{int(depth)}_g{int(grid)}"
            run_rmse_pipeline(
                repo_root=repo_root,
                out_dir=run_out,
                models_file=models_file,
                metadata_csv=metadata_csv,
                depth=int(depth),
                grid=int(grid),
                termination=float(termination),
                start_depth=int(start_depth),
                num_threads=int(num_threads),
                points_per_model=int(points_per_model),
                seeds=seeds,
                methods=["planar", "ours"],
                force=force,
                no_plot=True,
                verbose=verbose,
            )
            summary = read_csv_rows(run_out / "metrics_multiseed_summary.csv")
            for method in BASE_METHOD_ORDER:
                sr = find_method_row(summary, method)
                if sr is None:
                    continue
                rows.append(
                    {
                        "depth": int(depth),
                        "grid": int(grid),
                        "paper_method": method,
                        "seeds_count": int(to_float(sr.get("seeds_count", len(seeds)), len(seeds))),
                        "rmse_mean": to_float(sr.get("rmse_mean", float("nan"))),
                        "rmse_std": to_float(sr.get("rmse_std", float("nan"))),
                        "linf_mean": to_float(sr.get("linf_mean", float("nan"))),
                        "linf_std": to_float(sr.get("linf_std", float("nan"))),
                        "normal_rmse_deg_mean": to_float(sr.get("normal_rmse_deg_mean", float("nan"))),
                        "normal_rmse_deg_std": to_float(sr.get("normal_rmse_deg_std", float("nan"))),
                        "normal_p95_deg_mean": to_float(sr.get("normal_p95_deg_mean", float("nan"))),
                        "normal_p95_deg_std": to_float(sr.get("normal_p95_deg_std", float("nan"))),
                    }
                )

    rows.sort(
        key=lambda r: (
            int(r["depth"]),
            int(r["grid"]),
            BASE_METHOD_ORDER.index(str(r["paper_method"])),
        )
    )

    csv_path = exp_dir / "convergence_summary.csv"
    fields = [
        "depth",
        "grid",
        "paper_method",
        "seeds_count",
        "rmse_mean",
        "rmse_std",
        "linf_mean",
        "linf_std",
        "normal_rmse_deg_mean",
        "normal_rmse_deg_std",
        "normal_p95_deg_mean",
        "normal_p95_deg_std",
    ]
    write_csv(csv_path, rows, fields)
    write_convergence_table(rows, exp_dir / "table_exp1_convergence.tex")
    png = plot_convergence(rows, exp_dir / "convergence_rmse_normal.png")
    if png is not None:
        write_figure_tex(
            exp_dir / "fig_exp1_convergence.tex",
            png,
            "Convergence sweep of Surface-|SDF| RMSE and Normal RMSE under depth/grid settings (auto-generated).",
            "fig:exp123_convergence",
        )
    return rows


def load_metadata_map(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    rows = read_csv_rows(path)
    for r in rows:
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        pjson = str(r.get("param_json", "")).strip()
        params: Dict[str, float] = {}
        if pjson:
            try:
                obj = json.loads(pjson)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            params[str(k)] = float(v)
                        except Exception:
                            continue
            except Exception:
                pass
        out[name] = {
            "category": str(r.get("category", "")).strip(),
            "params": params,
        }
    return out


def hard_param_name_value(category: str, params: Dict[str, float]) -> Tuple[str, float]:
    if category == "cone":
        r = float(params.get("radius", 0.0))
        h = float(params.get("height", 0.0))
        return "h/r", (h / r) if r > 0.0 else float("nan")
    if category == "cylinder":
        r = float(params.get("radius", 0.0))
        h = float(params.get("height", 0.0))
        return "h/r", (h / r) if r > 0.0 else float("nan")
    if category == "torus":
        R = float(params.get("major_radius", 0.0))
        r = float(params.get("minor_radius", 0.0))
        return "r/R", (r / R) if R > 0.0 else float("nan")
    if category == "box":
        lx = float(params.get("length_x", 0.0))
        ly = float(params.get("length_y", 0.0))
        lz = float(params.get("length_z", 0.0))
        base = max(lx, ly)
        return "z/max(x,y)", (lz / base) if base > 0.0 else float("nan")
    return "param", float("nan")


def write_hardcase_table(rows: List[dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Hard-case parameter scan on analytic surface points (mean over seeds).}")
    lines.append("\\label{tab:exp123_hardcase_scan}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Category & Param & RMSE(Planar) & RMSE(Ours) & Gain(\\%) & NormalRMSE(Planar) & NormalRMSE(Ours) & Gain(\\%) & Models \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            f"{r['category']} & {r['param_desc']} & "
            f"{to_float(r['planar_rmse']):.6e} & {to_float(r['ours_rmse']):.6e} & {to_float(r['rmse_gain_pct']):.3f} & "
            f"{to_float(r['planar_normal_rmse']):.6e} & {to_float(r['ours_normal_rmse']):.6e} & {to_float(r['normal_rmse_gain_pct']):.3f} & "
            f"{int(to_float(r['count_models'], 0))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_hardcase(rows: List[dict], out_png: Path) -> Optional[Path]:
    if plt is None:
        return None
    if not rows:
        return None
    cats = sorted({str(r["category"]) for r in rows})
    if not cats:
        return None

    ncols = min(2, len(cats))
    nrows = int(math.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.6 * nrows), dpi=150)
    ax_list = np.asarray(axes).reshape(-1)

    for ax in ax_list[len(cats) :]:
        ax.axis("off")

    for idx, cat in enumerate(cats):
        ax = ax_list[idx]
        sub = [r for r in rows if str(r["category"]) == cat]
        sub.sort(key=lambda rr: float(rr["param_value_min"]))
        x = [float(rr["param_value_min"]) for rr in sub]
        y0 = [float(rr["planar_rmse"]) for rr in sub]
        y1 = [float(rr["ours_rmse"]) for rr in sub]
        ax.plot(x, y0, marker="o", label="Planar")
        ax.plot(x, y1, marker="o", label="Ours")
        ax.set_title(f"{cat} (Surface-|SDF| RMSE)")
        ax.set_xlabel(str(sub[0]["param_name"]) if sub else "param")
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_hardcase_experiment(
    repo_root: Path,
    out_dir: Path,
    models_file: Path,
    metadata_csv: Path,
    depth: int,
    grid: int,
    termination: float,
    start_depth: int,
    num_threads: int,
    points_per_model: int,
    seeds: Sequence[int],
    force: bool,
    verbose: bool,
) -> List[dict]:
    exp_dir = out_dir / "exp2_hardcase"
    run_rmse_pipeline(
        repo_root=repo_root,
        out_dir=exp_dir / "run",
        models_file=models_file,
        metadata_csv=metadata_csv,
        depth=depth,
        grid=grid,
        termination=termination,
        start_depth=start_depth,
        num_threads=num_threads,
        points_per_model=points_per_model,
        seeds=seeds,
        methods=["planar", "ours"],
        force=force,
        no_plot=True,
        verbose=verbose,
    )

    detailed = read_csv_rows(exp_dir / "run" / "metrics_detailed.csv")
    meta_map = load_metadata_map(metadata_csv)

    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in detailed:
        if str(r.get("status", "")) != "ok":
            continue
        m = str(r.get("paper_method", "")).strip()
        if m not in BASE_METHOD_ORDER:
            continue
        model = str(r.get("model_name", "")).strip()
        if not model:
            continue
        key = (model, m)
        bucket = grouped.setdefault(
            key,
            {"rmse": [], "linf": [], "normal_rmse_deg": [], "normal_p95_deg": []},
        )
        for k in ["rmse", "linf", "normal_rmse_deg", "normal_p95_deg"]:
            v = to_float(r.get(k, float("nan")))
            if math.isfinite(v):
                bucket[k].append(v)

    model_method_rows: List[dict] = []
    for (model, method), vals in grouped.items():
        mm = meta_map.get(model, {})
        cat = str(mm.get("category", ""))
        params = mm.get("params", {})
        if not isinstance(params, dict):
            params = {}
        param_name, param_value = hard_param_name_value(cat, params)
        model_method_rows.append(
            {
                "model_name": model,
                "category": cat,
                "param_name": param_name,
                "param_value": param_value,
                "paper_method": method,
                "rmse_mean": float(np.mean(vals["rmse"])) if vals["rmse"] else float("nan"),
                "linf_mean": float(np.mean(vals["linf"])) if vals["linf"] else float("nan"),
                "normal_rmse_mean": float(np.mean(vals["normal_rmse_deg"])) if vals["normal_rmse_deg"] else float("nan"),
                "normal_p95_mean": float(np.mean(vals["normal_p95_deg"])) if vals["normal_p95_deg"] else float("nan"),
            }
        )

    model_method_rows.sort(key=lambda r: (str(r["category"]), float(r["param_value"]), str(r["paper_method"])))
    write_csv(
        exp_dir / "hardcase_scan_model_method.csv",
        model_method_rows,
        [
            "model_name",
            "category",
            "param_name",
            "param_value",
            "paper_method",
            "rmse_mean",
            "linf_mean",
            "normal_rmse_mean",
            "normal_p95_mean",
        ],
    )

    per_category: Dict[str, Dict[str, List[float]]] = {}
    by_model: Dict[str, Dict[str, dict]] = {}
    for r in model_method_rows:
        by_model.setdefault(str(r["model_name"]), {})[str(r["paper_method"])] = r

    for _, mm in by_model.items():
        p = mm.get("Planar")
        o = mm.get("Ours")
        if p is None or o is None:
            continue
        cat = str(p["category"])
        key = per_category.setdefault(
            cat,
            {
                "param_values": [],
                "planar_rmse": [],
                "ours_rmse": [],
                "planar_normal_rmse": [],
                "ours_normal_rmse": [],
            },
        )
        key["param_values"].append(float(p["param_value"]))
        key["planar_rmse"].append(float(p["rmse_mean"]))
        key["ours_rmse"].append(float(o["rmse_mean"]))
        key["planar_normal_rmse"].append(float(p["normal_rmse_mean"]))
        key["ours_normal_rmse"].append(float(o["normal_rmse_mean"]))

    category_rows: List[dict] = []
    for cat, vals in per_category.items():
        p_rmse = np.asarray(vals["planar_rmse"], dtype=np.float64)
        o_rmse = np.asarray(vals["ours_rmse"], dtype=np.float64)
        p_nrmse = np.asarray(vals["planar_normal_rmse"], dtype=np.float64)
        o_nrmse = np.asarray(vals["ours_normal_rmse"], dtype=np.float64)
        param_values = np.asarray(vals["param_values"], dtype=np.float64)

        rmse_gain = ((p_rmse - o_rmse) / np.maximum(np.abs(p_rmse), 1e-20)) * 100.0
        nrmse_gain = ((p_nrmse - o_nrmse) / np.maximum(np.abs(p_nrmse), 1e-20)) * 100.0

        param_name = "param"
        if cat in ("cone", "cylinder"):
            param_name = "h/r"
        elif cat == "torus":
            param_name = "r/R"
        elif cat == "box":
            param_name = "z/max(x,y)"

        category_rows.append(
            {
                "category": cat,
                "param_name": param_name,
                "param_desc": f"{param_name} in [{np.min(param_values):.3f}, {np.max(param_values):.3f}]",
                "param_value_min": float(np.min(param_values)),
                "param_value_max": float(np.max(param_values)),
                "count_models": int(param_values.size),
                "planar_rmse": float(np.mean(p_rmse)),
                "ours_rmse": float(np.mean(o_rmse)),
                "rmse_gain_pct": float(np.mean(rmse_gain)),
                "planar_normal_rmse": float(np.mean(p_nrmse)),
                "ours_normal_rmse": float(np.mean(o_nrmse)),
                "normal_rmse_gain_pct": float(np.mean(nrmse_gain)),
            }
        )

    category_rows.sort(key=lambda r: str(r["category"]))
    write_csv(
        exp_dir / "hardcase_scan_category_summary.csv",
        category_rows,
        [
            "category",
            "param_name",
            "param_desc",
            "param_value_min",
            "param_value_max",
            "count_models",
            "planar_rmse",
            "ours_rmse",
            "rmse_gain_pct",
            "planar_normal_rmse",
            "ours_normal_rmse",
            "normal_rmse_gain_pct",
        ],
    )
    write_hardcase_table(category_rows, exp_dir / "table_exp2_hardcase_scan.tex")

    png = plot_hardcase(category_rows, exp_dir / "hardcase_scan_rmse.png")
    if png is not None:
        write_figure_tex(
            exp_dir / "fig_exp2_hardcase_scan.tex",
            png,
            "Hard-case parameter scan: category-wise Surface-|SDF| RMSE trend (auto-generated).",
            "fig:exp123_hardcase_scan",
        )
    return category_rows


def write_ablation_table(rows: List[dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study (termination-threshold sensitivity, mean$\\pm$std over seeds).}")
    lines.append("\\label{tab:exp123_ablation}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "Method & Surface-$|SDF|$ RMSE$\\downarrow$ & Surface-$|SDF|$ $L_\\infty\\downarrow$ & Normal RMSE$\\downarrow$ & Normal P95$\\downarrow$ & $\\Delta$RMSE vs Ours (\\%) \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            f"{r['method_label']} & "
            f"{format_mean_std(float(r['rmse_mean']), float(r['rmse_std']))} & "
            f"{format_mean_std(float(r['linf_mean']), float(r['linf_std']))} & "
            f"{format_mean_std(float(r['normal_rmse_mean']), float(r['normal_rmse_std']))} & "
            f"{format_mean_std(float(r['normal_p95_mean']), float(r['normal_p95_std']))} & "
            f"{float(r['delta_rmse_vs_ours_pct']):.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation_experiment(
    repo_root: Path,
    out_dir: Path,
    models_file: Path,
    metadata_csv: Path,
    depth: int,
    grid: int,
    start_depth: int,
    base_termination: float,
    ablation_termination: float,
    num_threads: int,
    points_per_model: int,
    seeds: Sequence[int],
    force: bool,
    verbose: bool,
) -> List[dict]:
    exp_dir = out_dir / "exp3_ablation"
    run_rmse_pipeline(
        repo_root=repo_root,
        out_dir=exp_dir / "base",
        models_file=models_file,
        metadata_csv=metadata_csv,
        depth=depth,
        grid=grid,
        termination=base_termination,
        start_depth=start_depth,
        num_threads=num_threads,
        points_per_model=points_per_model,
        seeds=seeds,
        methods=["planar", "ours"],
        force=force,
        no_plot=True,
        verbose=verbose,
    )
    run_rmse_pipeline(
        repo_root=repo_root,
        out_dir=exp_dir / "ours_ablation",
        models_file=models_file,
        metadata_csv=metadata_csv,
        depth=depth,
        grid=grid,
        termination=ablation_termination,
        start_depth=start_depth,
        num_threads=num_threads,
        points_per_model=points_per_model,
        seeds=seeds,
        methods=["ours"],
        force=force,
        no_plot=True,
        verbose=verbose,
    )

    base_rows = read_csv_rows(exp_dir / "base" / "metrics_multiseed_summary.csv")
    abl_rows = read_csv_rows(exp_dir / "ours_ablation" / "metrics_multiseed_summary.csv")

    plan = find_method_row(base_rows, "Planar")
    ours = find_method_row(base_rows, "Ours")
    abl = find_method_row(abl_rows, "Ours")
    if plan is None or ours is None or abl is None:
        raise RuntimeError("ablation summary missing expected rows")

    rows = [
        {
            "method_label": "Planar",
            "rmse_mean": to_float(plan.get("rmse_mean", float("nan"))),
            "rmse_std": to_float(plan.get("rmse_std", float("nan"))),
            "linf_mean": to_float(plan.get("linf_mean", float("nan"))),
            "linf_std": to_float(plan.get("linf_std", float("nan"))),
            "normal_rmse_mean": to_float(plan.get("normal_rmse_deg_mean", float("nan"))),
            "normal_rmse_std": to_float(plan.get("normal_rmse_deg_std", float("nan"))),
            "normal_p95_mean": to_float(plan.get("normal_p95_deg_mean", float("nan"))),
            "normal_p95_std": to_float(plan.get("normal_p95_deg_std", float("nan"))),
        },
        {
            "method_label": "Ours",
            "rmse_mean": to_float(ours.get("rmse_mean", float("nan"))),
            "rmse_std": to_float(ours.get("rmse_std", float("nan"))),
            "linf_mean": to_float(ours.get("linf_mean", float("nan"))),
            "linf_std": to_float(ours.get("linf_std", float("nan"))),
            "normal_rmse_mean": to_float(ours.get("normal_rmse_deg_mean", float("nan"))),
            "normal_rmse_std": to_float(ours.get("normal_rmse_deg_std", float("nan"))),
            "normal_p95_mean": to_float(ours.get("normal_p95_deg_mean", float("nan"))),
            "normal_p95_std": to_float(ours.get("normal_p95_deg_std", float("nan"))),
        },
        {
            "method_label": f"Ours-Abl($\\tau$={ablation_termination:.1e})",
            "rmse_mean": to_float(abl.get("rmse_mean", float("nan"))),
            "rmse_std": to_float(abl.get("rmse_std", float("nan"))),
            "linf_mean": to_float(abl.get("linf_mean", float("nan"))),
            "linf_std": to_float(abl.get("linf_std", float("nan"))),
            "normal_rmse_mean": to_float(abl.get("normal_rmse_deg_mean", float("nan"))),
            "normal_rmse_std": to_float(abl.get("normal_rmse_deg_std", float("nan"))),
            "normal_p95_mean": to_float(abl.get("normal_p95_deg_mean", float("nan"))),
            "normal_p95_std": to_float(abl.get("normal_p95_deg_std", float("nan"))),
        },
    ]

    ours_rmse = float(rows[1]["rmse_mean"])
    for r in rows:
        rr = float(r["rmse_mean"])
        r["delta_rmse_vs_ours_pct"] = ((rr - ours_rmse) / max(abs(ours_rmse), 1e-20)) * 100.0

    write_csv(
        exp_dir / "ablation_summary.csv",
        rows,
        [
            "method_label",
            "rmse_mean",
            "rmse_std",
            "linf_mean",
            "linf_std",
            "normal_rmse_mean",
            "normal_rmse_std",
            "normal_p95_mean",
            "normal_p95_std",
            "delta_rmse_vs_ours_pct",
        ],
    )
    write_ablation_table(rows, exp_dir / "table_exp3_ablation.tex")
    return rows


def write_manifest(path: Path, entries: List[Tuple[str, Path]]) -> None:
    lines = ["# exp123 outputs"]
    for name, p in entries:
        lines.append(f"{name}: {p}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-1 experiments 1+2+3 pipeline")
    parser.add_argument("--out_dir", default="output/paper1_exp123")
    parser.add_argument("--seeds", default="20260304,20260305,20260306,20260307,20260308")
    parser.add_argument("--convergence_depths", default="7,8,9")
    parser.add_argument("--convergence_grids", default="64,128,256,512")
    parser.add_argument("--points_per_model", type=int, default=20000)
    parser.add_argument("--start_depth", type=int, default=1)
    parser.add_argument("--base_depth", type=int, default=8)
    parser.add_argument("--base_grid", type=int, default=64)
    parser.add_argument("--base_termination", type=float, default=1e-3)
    parser.add_argument("--ablation_termination", type=float, default=1e-2)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_convergence", action="store_true")
    parser.add_argument("--skip_hardcase", action="store_true")
    parser.add_argument("--skip_ablation", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = resolve_path(repo_root, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds, [20260304])
    depths = parse_int_list(args.convergence_depths, [7, 8, 9])
    grids = parse_int_list(args.convergence_grids, [64, 128, 256, 512])
    if args.points_per_model <= 0:
        raise ValueError("--points_per_model must be positive")

    log(f"repo_root: {repo_root}")
    log(f"out_dir: {out_dir}")
    log(f"seeds: {seeds}")
    log(f"convergence depths: {depths}")
    log(f"convergence grids: {grids}")

    bench_full_dir = out_dir / "benchmarks_full_models"
    bench_full_manifest = bench_full_dir / "benchmark_models.txt"
    bench_full_meta = bench_full_dir / "benchmark_models_meta.csv"
    run_benchmark_generator(
        repo_root=repo_root,
        output_dir=bench_full_dir,
        profile="full",
        manifest=bench_full_manifest,
        metadata_csv=bench_full_meta,
        force=args.force,
        verbose=args.verbose,
    )

    bench_hard_dir = out_dir / "benchmarks_hard_models"
    bench_hard_manifest = bench_hard_dir / "benchmark_models.txt"
    bench_hard_meta = bench_hard_dir / "benchmark_models_meta.csv"
    run_benchmark_generator(
        repo_root=repo_root,
        output_dir=bench_hard_dir,
        profile="hard",
        manifest=bench_hard_manifest,
        metadata_csv=bench_hard_meta,
        force=args.force,
        verbose=args.verbose,
    )

    generated: List[Tuple[str, Path]] = []

    if not args.skip_convergence:
        run_convergence_experiment(
            repo_root=repo_root,
            out_dir=out_dir,
            models_file=bench_full_manifest,
            metadata_csv=bench_full_meta,
            depths=depths,
            grids=grids,
            seeds=seeds,
            start_depth=int(args.start_depth),
            termination=float(args.base_termination),
            num_threads=int(args.num_threads),
            points_per_model=int(args.points_per_model),
            force=args.force,
            verbose=args.verbose,
        )
        generated.extend(
            [
                ("exp1_csv", out_dir / "exp1_convergence" / "convergence_summary.csv"),
                ("exp1_table", out_dir / "exp1_convergence" / "table_exp1_convergence.tex"),
                ("exp1_fig", out_dir / "exp1_convergence" / "fig_exp1_convergence.tex"),
            ]
        )

    if not args.skip_hardcase:
        run_hardcase_experiment(
            repo_root=repo_root,
            out_dir=out_dir,
            models_file=bench_hard_manifest,
            metadata_csv=bench_hard_meta,
            depth=int(args.base_depth),
            grid=int(args.base_grid),
            termination=float(args.base_termination),
            start_depth=int(args.start_depth),
            num_threads=int(args.num_threads),
            points_per_model=int(args.points_per_model),
            seeds=seeds,
            force=args.force,
            verbose=args.verbose,
        )
        generated.extend(
            [
                ("exp2_csv", out_dir / "exp2_hardcase" / "hardcase_scan_category_summary.csv"),
                ("exp2_table", out_dir / "exp2_hardcase" / "table_exp2_hardcase_scan.tex"),
                ("exp2_fig", out_dir / "exp2_hardcase" / "fig_exp2_hardcase_scan.tex"),
            ]
        )

    if not args.skip_ablation:
        run_ablation_experiment(
            repo_root=repo_root,
            out_dir=out_dir,
            models_file=bench_full_manifest,
            metadata_csv=bench_full_meta,
            depth=int(args.base_depth),
            grid=int(args.base_grid),
            start_depth=int(args.start_depth),
            base_termination=float(args.base_termination),
            ablation_termination=float(args.ablation_termination),
            num_threads=int(args.num_threads),
            points_per_model=int(args.points_per_model),
            seeds=seeds,
            force=args.force,
            verbose=args.verbose,
        )
        generated.extend(
            [
                ("exp3_csv", out_dir / "exp3_ablation" / "ablation_summary.csv"),
                ("exp3_table", out_dir / "exp3_ablation" / "table_exp3_ablation.tex"),
            ]
        )

    write_manifest(out_dir / "exp123_manifest.txt", generated)
    log(f"done. outputs listed in: {out_dir / 'exp123_manifest.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
