#!/usr/bin/env python3
"""
Analytic point-sampling evaluation for benchmark primitives.

This module provides:
- benchmark metadata loading (category + param_json)
- analytic SDF for sphere/cube/box/cylinder/cone/torus
- analytic surface-point sampling
- trilinear RAW-grid query and finite-difference gradients
- surface-point |SDF-0| metrics and normal-angle metrics
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AnalyticModelSpec:
    name: str
    category: str
    params: Dict[str, float]


def load_analytic_metadata(path: Path) -> Dict[str, AnalyticModelSpec]:
    if not path.exists():
        raise FileNotFoundError(f"metadata csv not found: {path}")
    out: Dict[str, AnalyticModelSpec] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("name", "")).strip()
            cat = str(row.get("category", "")).strip()
            if not name or not cat:
                continue
            params: Dict[str, float] = {}
            pjson = str(row.get("param_json", "")).strip()
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
            out[name] = AnalyticModelSpec(name=name, category=cat, params=params)
    return out


def sample_points_in_bbox(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    inset_ratio: float = 1e-4,
) -> np.ndarray:
    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)
    ext = np.maximum(bmax - bmin, 1e-12)
    inset = inset_ratio * ext
    lo = bmin + inset
    hi = bmax - inset
    return rng.uniform(lo, hi, size=(int(n_points), 3))


def _sdf_box(points: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    b = np.array([hx, hy, hz], dtype=np.float64)
    q = np.abs(p) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _sdf_sphere(points: np.ndarray, radius: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return np.linalg.norm(p, axis=1) - float(radius)


def _sdf_torus(points: np.ndarray, major_radius: float, minor_radius: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    rho = np.linalg.norm(p[:, :2], axis=1)
    qx = rho - float(major_radius)
    qy = p[:, 2]
    return np.sqrt(qx * qx + qy * qy) - float(minor_radius)


def _sdf_capped_cylinder(points: np.ndarray, radius: float, height: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    r = float(radius)
    h2 = 0.5 * float(height)
    rho = np.linalg.norm(p[:, :2], axis=1)
    d0 = rho - r
    d1 = np.abs(p[:, 2]) - h2
    outside = np.sqrt(np.maximum(d0, 0.0) ** 2 + np.maximum(d1, 0.0) ** 2)
    inside = np.minimum(np.maximum(d0, d1), 0.0)
    return outside + inside


def _sdf_cone_apex_z(points: np.ndarray, radius: float, height: float) -> np.ndarray:
    """
    Finite cone:
    - apex at (0,0,height)
    - base disk centered at z=0 with radius
    - axis along +z
    """
    p = np.asarray(points, dtype=np.float64)
    r = float(radius)
    h = float(height)
    eps = 1e-12
    rho = np.linalg.norm(p[:, :2], axis=1)
    z = p[:, 2]

    # Side surface distance via 2D segment distance in (rho,z):
    # segment endpoints: A=(0,h), B=(r,0)
    ax, ay = 0.0, h
    bx, by = r, 0.0
    vx = bx - ax
    vy = by - ay
    vv = vx * vx + vy * vy + eps
    wx = rho - ax
    wy = z - ay
    t = np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0)
    cx = ax + t * vx
    cy = ay + t * vy
    d_side = np.sqrt((rho - cx) ** 2 + (z - cy) ** 2)

    # Base disk distance (z=0, rho<=r)
    d_base = np.where(rho <= r, np.abs(z), np.sqrt((rho - r) ** 2 + z * z))

    dist = np.minimum(d_side, d_base)

    # Inside test
    # local radius at z in [0,h]: r(z)=r*(1-z/h)
    rz = r * np.maximum(0.0, 1.0 - z / max(h, eps))
    inside = (z >= 0.0) & (z <= h) & (rho <= rz + 1e-12)
    return np.where(inside, -dist, dist)


def analytic_sdf(spec: AnalyticModelSpec, points: np.ndarray) -> np.ndarray:
    cat = spec.category
    p = spec.params
    if cat == "sphere":
        return _sdf_sphere(points, p["radius"])
    if cat == "cube":
        h = p["half_size"]
        return _sdf_box(points, h, h, h)
    if cat == "box":
        return _sdf_box(points, 0.5 * p["length_x"], 0.5 * p["length_y"], 0.5 * p["length_z"])
    if cat == "cylinder":
        return _sdf_capped_cylinder(points, p["radius"], p["height"])
    if cat == "cone":
        return _sdf_cone_apex_z(points, p["radius"], p["height"])
    if cat == "torus":
        return _sdf_torus(points, p["major_radius"], p["minor_radius"])
    raise ValueError(f"unsupported analytic category: {cat}")


def _sample_box_surface(
    hx: float,
    hy: float,
    hz: float,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n_points)
    if n <= 0:
        raise ValueError("n_points must be positive")

    # Paired-face areas (both signs): +/-X, +/-Y, +/-Z
    w = np.asarray(
        [
            8.0 * hy * hz,
            8.0 * hx * hz,
            8.0 * hx * hy,
        ],
        dtype=np.float64,
    )
    if np.any(w <= 0.0):
        raise ValueError("invalid box half sizes for surface sampling")
    w = w / np.sum(w)

    axis = rng.choice(3, size=n, p=w)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    pts = np.zeros((n, 3), dtype=np.float64)
    nrm = np.zeros((n, 3), dtype=np.float64)

    m0 = axis == 0
    if np.any(m0):
        k = int(np.count_nonzero(m0))
        pts[m0, 0] = sign[m0] * hx
        pts[m0, 1] = rng.uniform(-hy, hy, size=k)
        pts[m0, 2] = rng.uniform(-hz, hz, size=k)
        nrm[m0, 0] = sign[m0]

    m1 = axis == 1
    if np.any(m1):
        k = int(np.count_nonzero(m1))
        pts[m1, 0] = rng.uniform(-hx, hx, size=k)
        pts[m1, 1] = sign[m1] * hy
        pts[m1, 2] = rng.uniform(-hz, hz, size=k)
        nrm[m1, 1] = sign[m1]

    m2 = axis == 2
    if np.any(m2):
        k = int(np.count_nonzero(m2))
        pts[m2, 0] = rng.uniform(-hx, hx, size=k)
        pts[m2, 1] = rng.uniform(-hy, hy, size=k)
        pts[m2, 2] = sign[m2] * hz
        nrm[m2, 2] = sign[m2]

    return pts, nrm


def _sample_cylinder_surface(
    radius: float,
    height: float,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n_points)
    if n <= 0:
        raise ValueError("n_points must be positive")
    r = float(radius)
    h = float(height)
    h2 = 0.5 * h

    side_area = 2.0 * np.pi * r * h
    cap_area = np.pi * r * r
    w = np.asarray([side_area, cap_area, cap_area], dtype=np.float64)
    w = w / np.sum(w)
    comp = rng.choice(3, size=n, p=w)  # 0:side, 1:top, 2:bottom

    pts = np.zeros((n, 3), dtype=np.float64)
    nrm = np.zeros((n, 3), dtype=np.float64)

    ms = comp == 0
    if np.any(ms):
        k = int(np.count_nonzero(ms))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=k)
        c = np.cos(theta)
        s = np.sin(theta)
        pts[ms, 0] = r * c
        pts[ms, 1] = r * s
        pts[ms, 2] = rng.uniform(-h2, h2, size=k)
        nrm[ms, 0] = c
        nrm[ms, 1] = s

    mt = comp == 1
    if np.any(mt):
        k = int(np.count_nonzero(mt))
        rho = r * np.sqrt(rng.random(k))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=k)
        pts[mt, 0] = rho * np.cos(theta)
        pts[mt, 1] = rho * np.sin(theta)
        pts[mt, 2] = h2
        nrm[mt, 2] = 1.0

    mb = comp == 2
    if np.any(mb):
        k = int(np.count_nonzero(mb))
        rho = r * np.sqrt(rng.random(k))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=k)
        pts[mb, 0] = rho * np.cos(theta)
        pts[mb, 1] = rho * np.sin(theta)
        pts[mb, 2] = -h2
        nrm[mb, 2] = -1.0

    return pts, nrm


def _sample_cone_surface_apex_z(
    radius: float,
    height: float,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite cone surface sampler:
    - apex at (0,0,height)
    - base disk centered at z=0
    """
    n = int(n_points)
    if n <= 0:
        raise ValueError("n_points must be positive")
    r = float(radius)
    h = float(height)
    sl = float(np.sqrt(r * r + h * h))
    side_area = np.pi * r * sl
    base_area = np.pi * r * r
    p_side = side_area / max(side_area + base_area, 1e-20)
    choose_side = rng.random(n) < p_side

    pts = np.zeros((n, 3), dtype=np.float64)
    nrm = np.zeros((n, 3), dtype=np.float64)

    ms = choose_side
    if np.any(ms):
        k = int(np.count_nonzero(ms))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=k)
        t = np.sqrt(np.clip(rng.random(k), 1e-12, 1.0))
        rho = r * t
        z = h * (1.0 - t)
        c = np.cos(theta)
        s = np.sin(theta)
        pts[ms, 0] = rho * c
        pts[ms, 1] = rho * s
        pts[ms, 2] = z
        # Side outward normal (constant slope in (rho,z))
        nrm[ms, 0] = (h / sl) * c
        nrm[ms, 1] = (h / sl) * s
        nrm[ms, 2] = (r / sl)

    mb = ~ms
    if np.any(mb):
        k = int(np.count_nonzero(mb))
        rho = r * np.sqrt(rng.random(k))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=k)
        pts[mb, 0] = rho * np.cos(theta)
        pts[mb, 1] = rho * np.sin(theta)
        pts[mb, 2] = 0.0
        nrm[mb, 2] = -1.0

    return pts, nrm


def _sample_torus_surface(
    major_radius: float,
    minor_radius: float,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n_points)
    if n <= 0:
        raise ValueError("n_points must be positive")
    R = float(major_radius)
    r = float(minor_radius)

    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = np.empty((n,), dtype=np.float64)
    vmax = max(R + abs(r), 1e-20)
    filled = 0
    while filled < n:
        k = max(32, 2 * (n - filled))
        cand = rng.uniform(0.0, 2.0 * np.pi, size=k)
        w = R + r * np.cos(cand)
        keep = (w > 0.0) & (rng.random(k) <= np.clip(w / vmax, 0.0, 1.0))
        if not np.any(keep):
            continue
        take = cand[keep]
        m = min(n - filled, take.size)
        v[filled : filled + m] = take[:m]
        filled += m

    cu = np.cos(u)
    su = np.sin(u)
    cv = np.cos(v)
    sv = np.sin(v)
    rr = R + r * cv

    pts = np.stack([rr * cu, rr * su, r * sv], axis=1)
    nrm = np.stack([cu * cv, su * cv, sv], axis=1)
    return pts, nrm


def sample_surface_points(
    spec: AnalyticModelSpec,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    cat = spec.category
    p = spec.params
    n = int(n_points)
    if n <= 0:
        raise ValueError("n_points must be positive")

    if cat == "sphere":
        r = float(p["radius"])
        z = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        x = r * xy * np.cos(phi)
        y = r * xy * np.sin(phi)
        zz = r * z
        pts = np.stack([x, y, zz], axis=1)
        nrm = pts / max(r, 1e-20)
        return pts, nrm

    if cat == "cube":
        h = float(p["half_size"])
        return _sample_box_surface(h, h, h, n, rng)

    if cat == "box":
        return _sample_box_surface(
            0.5 * float(p["length_x"]),
            0.5 * float(p["length_y"]),
            0.5 * float(p["length_z"]),
            n,
            rng,
        )

    if cat == "cylinder":
        return _sample_cylinder_surface(float(p["radius"]), float(p["height"]), n, rng)

    if cat == "cone":
        return _sample_cone_surface_apex_z(float(p["radius"]), float(p["height"]), n, rng)

    if cat == "torus":
        return _sample_torus_surface(float(p["major_radius"]), float(p["minor_radius"]), n, rng)

    raise ValueError(f"unsupported analytic category: {cat}")


def gradient_fd_values(
    value_fn,
    points: np.ndarray,
    step: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    h = float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")
    grad = np.zeros((pts.shape[0], 3), dtype=np.float64)
    for axis in range(3):
        off = np.zeros((1, 3), dtype=np.float64)
        off[0, axis] = h
        vp = value_fn(pts + off)
        vm = value_fn(pts - off)
        grad[:, axis] = (vp - vm) / (2.0 * h)
    return grad


def sample_raw_trilinear(
    res: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)

    ext = np.maximum(bmax - bmin, 1e-12)
    u = (p - bmin) / ext * float(res - 1)
    u = np.clip(u, 0.0, float(res - 1) - 1e-9)

    i0 = np.floor(u).astype(np.int64)
    t = u - i0
    i1 = np.minimum(i0 + 1, res - 1)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    c000 = g[x0, y0, z0]
    c100 = g[x1, y0, z0]
    c010 = g[x0, y1, z0]
    c110 = g[x1, y1, z0]
    c001 = g[x0, y0, z1]
    c101 = g[x1, y0, z1]
    c011 = g[x0, y1, z1]
    c111 = g[x1, y1, z1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


def gradient_raw_fd(
    res: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid: np.ndarray,
    points: np.ndarray,
    step: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    h = float(step)
    grad = np.zeros((pts.shape[0], 3), dtype=np.float64)

    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)
    eps = 1e-12
    lo = bmin + h + eps
    hi = bmax - h - eps
    q = np.clip(pts, lo, hi)

    for axis in range(3):
        off = np.zeros((1, 3), dtype=np.float64)
        off[0, axis] = h
        vp = sample_raw_trilinear(res, bmin, bmax, grid, q + off)
        vm = sample_raw_trilinear(res, bmin, bmax, grid, q - off)
        grad[:, axis] = (vp - vm) / (2.0 * h)
    return grad


def evaluate_analytic_points_against_raw(
    spec: AnalyticModelSpec,
    res: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid: np.ndarray,
    n_points: int,
    seed: int,
    grad_step_world: Optional[float] = None,
    narrowband: Optional[float] = None,
) -> Dict[str, float]:
    """
    Surface-point validation:
    - points are sampled on analytic model surface
    - SDF reference is exactly zero at these points
    - normal reference comes from analytic surface normals
    """
    if int(n_points) <= 0:
        raise ValueError("n_points must be positive")
    rng = np.random.default_rng(int(seed))
    points, ref_nrm = sample_surface_points(spec, n_points, rng)

    ext = np.asarray(bbox_max, dtype=np.float64) - np.asarray(bbox_min, dtype=np.float64)
    cell = float(np.max(ext) / max(1, res - 1))
    h = grad_step_world if grad_step_world is not None and grad_step_world > 0.0 else 0.6 * cell

    pred_sdf = sample_raw_trilinear(res, bbox_min, bbox_max, grid, points)
    # Surface points are the reference zero level set: SDF_ref == 0.
    if narrowband is not None and float(narrowband) > 0.0:
        # Keep for backward CLI compatibility; no-op for surface sampling.
        _ = narrowband

    diff = pred_sdf
    rmse = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))
    pred_grad = gradient_raw_fd(res, bbox_min, bbox_max, grid, points, h)

    ref_norm = np.linalg.norm(ref_nrm, axis=1)
    pred_norm = np.linalg.norm(pred_grad, axis=1)
    valid = (ref_norm > 1e-10) & (pred_norm > 1e-10) & np.all(np.isfinite(pred_grad), axis=1)
    valid_count = int(np.count_nonzero(valid))
    if valid_count > 0:
        rg = ref_nrm[valid] / ref_norm[valid][:, None]
        pg = pred_grad[valid] / pred_norm[valid][:, None]
        cosv = np.sum(rg * pg, axis=1)
        cosv = np.clip(cosv, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosv))
        normal_mean_deg = float(np.mean(ang))
        normal_rmse_deg = float(np.sqrt(np.mean(ang * ang)))
        normal_p95_deg = float(np.percentile(ang, 95.0))
    else:
        normal_mean_deg = float("nan")
        normal_rmse_deg = float("nan")
        normal_p95_deg = float("nan")

    return {
        "rmse": rmse,
        "linf": linf,
        "num_samples": int(points.shape[0]),
        "normal_valid": valid_count,
        "normal_mean_deg": normal_mean_deg,
        "normal_rmse_deg": normal_rmse_deg,
        "normal_p95_deg": normal_p95_deg,
    }
