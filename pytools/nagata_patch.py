"""
Nagata Patch计算模块
基于 Nagata (2005) 插值算法

参考: Matlab实现 (temps/ComputeCurvature.m, NagataPatch.m, PlotNagataPatch.m)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# 角度容差: 当法向量夹角小于0.1度时，退化为线性插值
ANGLE_TOL = np.cos(0.1 * np.pi / 180)


def compute_curvature(d: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """
    计算Nagata插值的曲率系数向量
    
    基于Nagata 2005论文的曲率系数计算方法。
    当两个法向量接近平行时，退化为线性插值（返回零向量）。
    
    Args:
        d: 方向向量 (x1 - x0), shape (3,)
        n0: 起点法向量, shape (3,)
        n1: 终点法向量, shape (3,)
        
    Returns:
        cvec: 曲率系数向量, shape (3,)
    """
    # 3D情况的几何方法计算
    v = 0.5 * (n0 + n1)          # 法向量平均
    delta_v = 0.5 * (n0 - n1)    # 法向量差
    
    dv = np.dot(d, v)            # d在v方向的投影
    d_delta_v = np.dot(d, delta_v)  # d在delta_v方向的投影
    
    delta_c = np.dot(n0, delta_v)   # 法向量相关性
    c = 1 - 2 * delta_c             # 角度相关系数
    
    # 检查法向量是否接近平行
    if abs(c) > ANGLE_TOL:
        # 法向量接近平行，退化为线性插值
        return np.zeros(3)
    
    # 避免除零
    denom1 = 1 - delta_c
    denom2 = delta_c
    
    if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
        return np.zeros(3)
    
    cvec = (d_delta_v / denom1) * v + (dv / denom2) * delta_v
    return cvec


def nagata_patch(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    n00: np.ndarray, n10: np.ndarray, n11: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算Nagata曲面片的三个曲率系数
    
    对于三角形的三条边，分别计算曲率系数:
    - c1: 边 x00 -> x10
    - c2: 边 x10 -> x11  
    - c3: 边 x00 -> x11
    
    Args:
        x00, x10, x11: 三角形三个顶点坐标, shape (3,)
        n00, n10, n11: 三个顶点的法向量, shape (3,)
        
    Returns:
        c1, c2, c3: 三条边的曲率系数向量, shape (3,)
    """
    c1 = compute_curvature(x10 - x00, n00, n10)
    c2 = compute_curvature(x11 - x10, n10, n11)
    c3 = compute_curvature(x11 - x00, n00, n11)
    
    return c1, c2, c3


def evaluate_nagata_patch(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
    u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """
    在参数域采样点计算Nagata曲面坐标
    
    Nagata曲面参数方程:
    x(u,v) = x00*(1-u) + x10*(u-v) + x11*v 
           - c1*(1-u)*(u-v) - c2*(u-v)*v - c3*(1-u)*v
           
    参数域: u,v ∈ [0,1] 且 v ≤ u (三角形区域)
    
    Args:
        x00, x10, x11: 三角形顶点坐标, shape (3,)
        c1, c2, c3: 曲率系数, shape (3,)
        u, v: 参数坐标, 可以是标量或数组
        
    Returns:
        points: 曲面采样点坐标, shape (..., 3)
    """
    # 确保u, v是数组
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    
    # 计算各项
    one_minus_u = 1 - u
    u_minus_v = u - v
    
    # 线性项
    linear = (x00[:, None] * one_minus_u + 
              x10[:, None] * u_minus_v + 
              x11[:, None] * v)
    
    # 二次修正项
    quadratic = (c1[:, None] * (one_minus_u * u_minus_v) +
                 c2[:, None] * (u_minus_v * v) +
                 c3[:, None] * (one_minus_u * v))
    
    # 最终坐标
    points = linear - quadratic
    
    return points.T  # shape (..., 3)


def sample_nagata_triangle(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    n00: np.ndarray, n10: np.ndarray, n11: np.ndarray,
    resolution: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个三角形采样Nagata曲面，返回点云和三角形面片
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        n00, n10, n11: 顶点法向量
        resolution: 参数域采样密度 (M x M 网格)
        
    Returns:
        vertices: 采样点坐标, shape (N, 3)
        faces: 三角形索引, shape (M, 3)
    """
    # 计算曲率系数
    c1, c2, c3 = nagata_patch(x00, x10, x11, n00, n10, n11)
    
    # 在参数域采样 (只取下三角区域 v <= u)
    u_vals = np.linspace(0, 1, resolution)
    v_vals = np.linspace(0, 1, resolution)
    
    # 收集所有采样点
    vertices = []
    vertex_map = {}  # (i, j) -> vertex_index
    
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            if v <= u + 1e-10:  # 下三角区域
                point = evaluate_nagata_patch(x00, x10, x11, c1, c2, c3, u, v)
                vertex_map[(i, j)] = len(vertices)
                vertices.append(point.flatten())
    
    vertices = np.array(vertices)
    
    # 生成三角形面片
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # 检查是否在有效区域内
            if (i, j) in vertex_map and (i+1, j) in vertex_map and (i+1, j+1) in vertex_map:
                # 下三角形
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j)], 
                            vertex_map[(i+1, j+1)]])
            
            if (i, j) in vertex_map and (i+1, j+1) in vertex_map and (i, j+1) in vertex_map:
                # 上三角形 (如果在有效区域)
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j+1)], 
                            vertex_map[(i, j+1)]])
    
    faces = np.array(faces) if faces else np.zeros((0, 3), dtype=int)
    
    return vertices, faces


def sample_all_nagata_patches(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    resolution: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对整个网格的所有三角形采样Nagata曲面
    
    Args:
        vertices: 网格顶点, shape (num_vertices, 3)
        triangles: 三角形索引, shape (num_triangles, 3)
        tri_vertex_normals: 每个三角形的顶点法向量, shape (num_triangles, 3, 3)
        resolution: 每个三角形的采样密度
        
    Returns:
        all_vertices: 所有采样点, shape (N, 3)
        all_faces: 所有三角形面片, shape (M, 3)
        face_to_original: 每个采样三角形对应的原始三角形索引
    """
    all_vertices = []
    all_faces = []
    face_to_original = []
    
    vertex_offset = 0
    
    for tri_idx in range(triangles.shape[0]):
        # 获取三角形顶点
        i00, i10, i11 = triangles[tri_idx]
        x00 = vertices[i00]
        x10 = vertices[i10]
        x11 = vertices[i11]
        
        # 获取顶点法向量
        n00 = tri_vertex_normals[tri_idx, 0]
        n10 = tri_vertex_normals[tri_idx, 1]
        n11 = tri_vertex_normals[tri_idx, 2]
        
        # 采样该三角形
        tri_verts, tri_faces = sample_nagata_triangle(
            x00, x10, x11, n00, n10, n11, resolution
        )
        
        if len(tri_verts) > 0:
            all_vertices.append(tri_verts)
            all_faces.append(tri_faces + vertex_offset)
            face_to_original.extend([tri_idx] * len(tri_faces))
            vertex_offset += len(tri_verts)
    
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        all_faces = np.vstack(all_faces)
        face_to_original = np.array(face_to_original)
    else:
        all_vertices = np.zeros((0, 3))
        all_faces = np.zeros((0, 3), dtype=int)
        face_to_original = np.array([], dtype=int)
    
    return all_vertices, all_faces, face_to_original


# =============================================================================
# 折痕裂隙修复相关函数 (Crease-aware Nagata patch)
# =============================================================================

def compute_crease_direction(n_L: np.ndarray, n_R: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    计算折痕切向单位方向
    
    折痕方向位于两侧切平面的交线上，即两侧法向的叉积方向。
    
    Args:
        n_L: 左侧法向量, shape (3,)
        n_R: 右侧法向量, shape (3,)
        e: 边向量 (B - A), shape (3,)
        
    Returns:
        d: 折痕切向单位方向, shape (3,)
    """
    cross = np.cross(n_L, n_R)
    norm = np.linalg.norm(cross)
    
    if norm < 1e-10:
        # 退化情况：法向量近乎平行，使用边方向
        d = e / np.linalg.norm(e)
    else:
        d = cross / norm
    
    # 确保方向与边向量同向
    if np.dot(d, e) < 0:
        d = -d
    
    return d


def compute_c_sharp(A: np.ndarray, B: np.ndarray, 
                    d_A: np.ndarray, d_B: np.ndarray,
                    reg_lambda: float = 1e-6,
                    kappa: float = 2.0) -> np.ndarray:
    """
    计算共享边界系数 c^{sharp}
    
    通过最小二乘求解满足二次边界约束的端点切向长度。
    
    Args:
        A: 端点A坐标, shape (3,)
        B: 端点B坐标, shape (3,)
        d_A: 端点A的折痕切向单位方向, shape (3,)
        d_B: 端点B的折痕切向单位方向, shape (3,)
        reg_lambda: 正则化参数
        kappa: 过冲钳制系数
        
    Returns:
        c_sharp: 共享边界系数, shape (3,)
    """
    e = B - A
    e_norm = np.linalg.norm(e)
    
    # 构建 2x2 Gram 矩阵
    G = np.array([
        [np.dot(d_A, d_A), np.dot(d_A, d_B)],
        [np.dot(d_A, d_B), np.dot(d_B, d_B)]
    ])
    
    # 右端项
    r = np.array([
        2 * np.dot(e, d_A),
        2 * np.dot(e, d_B)
    ])
    
    # 正则化求解
    G_reg = G + reg_lambda * np.eye(2)
    try:
        ell = np.linalg.solve(G_reg, r)
    except np.linalg.LinAlgError:
        # 矩阵奇异，回退
        ell = np.array([e_norm, e_norm])
    
    ell_A, ell_B = ell
    T_A = ell_A * d_A
    T_B = ell_B * d_B
    
    # 计算 c^sharp
    c_sharp = (T_B - T_A) / 2
    
    # 过冲钳制
    c_norm = np.linalg.norm(c_sharp)
    max_c = kappa * e_norm
    if c_norm > max_c:
        c_sharp = c_sharp * (max_c / c_norm)
    
    return c_sharp


def smoothstep(t: np.ndarray) -> np.ndarray:
    """
    五次光滑过渡函数
    
    w = 6t^5 - 15t^4 + 10t^3
    满足 w(0)=0, w(1)=1, w'(0)=w'(1)=0
    
    Args:
        t: 输入参数, 可以是标量或数组, 应在 [0, 1] 范围内
        
    Returns:
        w: 过渡值
    """
    t = np.clip(t, 0, 1)
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def evaluate_nagata_patch_with_crease(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1_orig: np.ndarray, c2_orig: np.ndarray, c3_orig: np.ndarray,
    c1_sharp: np.ndarray, c2_sharp: np.ndarray, c3_sharp: np.ndarray,
    is_crease: tuple,  # (bool, bool, bool) for edges 1, 2, 3
    u: np.ndarray, v: np.ndarray,
    d0: float = 0.1
) -> np.ndarray:
    """
    带折痕修复的 Nagata 曲面求值
    
    对标记为裂隙边的 c_i，根据到边距离进行融合。
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        c1_orig, c2_orig, c3_orig: 原始曲率系数
        c1_sharp, c2_sharp, c3_sharp: 裂隙修复系数
        is_crease: 三条边是否为裂隙边
        u, v: 参数坐标
        d0: 融合宽度参数
        
    Returns:
        points: 曲面采样点坐标
    """
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    
    # 计算到各边的距离参数
    # 边1: v=0, d1=v
    # 边2: u=1, d2=1-u
    # 边3: u=v, d3=u-v
    d1 = v
    d2 = 1 - u
    d3 = u - v
    
    # 计算有效的 c_i^eff
    def blend_c(c_orig, c_sharp, is_cr, d):
        if not is_cr:
            return c_orig
        s = np.clip(d / d0, 0, 1)
        w = smoothstep(s)
        # w=0 时用 c_sharp, w=1 时用 c_orig
        return (1 - w)[:, None] * c_sharp + w[:, None] * c_orig
    
    # 对每个采样点计算有效系数
    c1_eff = blend_c(c1_orig, c1_sharp, is_crease[0], d1)
    c2_eff = blend_c(c2_orig, c2_sharp, is_crease[1], d2)
    c3_eff = blend_c(c3_orig, c3_sharp, is_crease[2], d3)
    
    # 线性项
    one_minus_u = 1 - u
    u_minus_v = u - v
    
    linear = (x00 * one_minus_u[:, None] + 
              x10 * u_minus_v[:, None] + 
              x11 * v[:, None])
    
    # 二次修正项 (每个点有不同的 c_eff)
    quadratic = (c1_eff * (one_minus_u * u_minus_v)[:, None] +
                 c2_eff * (u_minus_v * v)[:, None] +
                 c3_eff * (one_minus_u * v)[:, None])
    
    points = linear - quadratic
    return points


def sample_nagata_triangle_with_crease(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    n00: np.ndarray, n10: np.ndarray, n11: np.ndarray,
    c_sharps: dict,  # {edge_key: c_sharp}
    edge_keys: tuple,  # 三条边的全局键
    resolution: int = 10,
    d0: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    带折痕修复的单三角形 Nagata 采样
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        n00, n10, n11: 顶点法向量
        c_sharps: 裂隙边的共享系数字典
        edge_keys: 三条边的全局键 (边1, 边2, 边3)
        resolution: 采样密度
        d0: 融合宽度
        
    Returns:
        vertices, faces: 采样结果
    """
    # 计算原始曲率系数
    c1_orig, c2_orig, c3_orig = nagata_patch(x00, x10, x11, n00, n10, n11)
    
    # 获取裂隙修复系数
    is_crease = [False, False, False]
    c1_sharp = c1_orig.copy()
    c2_sharp = c2_orig.copy()
    c3_sharp = c3_orig.copy()
    
    if edge_keys[0] in c_sharps:
        is_crease[0] = True
        c1_sharp = c_sharps[edge_keys[0]]
    if edge_keys[1] in c_sharps:
        is_crease[1] = True
        c2_sharp = c_sharps[edge_keys[1]]
    if edge_keys[2] in c_sharps:
        is_crease[2] = True
        c3_sharp = c_sharps[edge_keys[2]]
    
    # 生成参数域采样点
    u_vals = np.linspace(0, 1, resolution)
    v_vals = np.linspace(0, 1, resolution)
    
    vertices = []
    vertex_map = {}
    
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            if v <= u + 1e-10:
                point = evaluate_nagata_patch_with_crease(
                    x00, x10, x11,
                    c1_orig, c2_orig, c3_orig,
                    c1_sharp, c2_sharp, c3_sharp,
                    tuple(is_crease),
                    np.array([u]), np.array([v]),
                    d0
                )
                vertex_map[(i, j)] = len(vertices)
                vertices.append(point.flatten())
    
    vertices = np.array(vertices) if vertices else np.zeros((0, 3))
    
    # 生成面片
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            if (i, j) in vertex_map and (i+1, j) in vertex_map and (i+1, j+1) in vertex_map:
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j)], 
                            vertex_map[(i+1, j+1)]])
            
            if (i, j) in vertex_map and (i+1, j+1) in vertex_map and (i, j+1) in vertex_map:
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j+1)], 
                            vertex_map[(i, j+1)]])
    
    faces = np.array(faces) if faces else np.zeros((0, 3), dtype=int)
    
    return vertices, faces





# =============================================================================
# 投影与查询逻辑 (Projection & Query)
# =============================================================================

def smoothstep_deriv(t: np.ndarray) -> np.ndarray:
    """五次光滑过渡函数的导数: w'(t) = 30t^2(t-1)^2"""
    t = np.clip(t, 0, 1)
    term = t * (t - 1.0)
    return 30.0 * term * term

def evaluate_nagata_derivatives(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
    u: float, v: float,
    # 可选: 折痕相关参数
    is_crease: tuple = (False, False, False),
    c_sharps: tuple = (None, None, None), 
    d0: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Nagata 曲面的一阶偏导数 (dX/du, dX/dv)
    支持折痕融合逻辑
    """
    # 基础几何导数 (线性部分)
    # x(u,v)_linear = x00(1-u) + x10(u-v) + x11*v
    # dLin/du = -x00 + x10
    # dLin/dv = -x10 + x11
    dLin_du = -x00 + x10
    dLin_dv = -x10 + x11
    
    # 二次项系数混合处理
    # 距离参数定义
    d_params = [v, 1.0 - u, u - v] # d1(v), d2(1-u), d3(u-v)
    
    # 距离对 (u,v) 的导数 [dd/du, dd/dv]
    dd_du = [0.0, -1.0, 1.0]
    dd_dv = [1.0, 0.0, -1.0]
    
    # 准备系数及其导数
    coeffs = [c1, c2, c3]
    # Handle None in c_sharps
    sharp_coeffs = [c if s is None else s for c, s in zip(coeffs, c_sharps)]
    
    c_eff = []
    dc_du = []
    dc_dv = []
    
    for i in range(3):
        if not is_crease[i]:
            c_eff.append(coeffs[i])
            dc_du.append(np.zeros(3))
            dc_dv.append(np.zeros(3))
        else:
            dist = d_params[i]
            c_orig = coeffs[i]
            c_sharp = sharp_coeffs[i]
            
            if dist <= 0:
                c_eff.append(c_sharp)
                dc_du.append(np.zeros(3))
                dc_dv.append(np.zeros(3))
            elif dist >= d0:
                c_eff.append(c_orig)
                dc_du.append(np.zeros(3))
                dc_dv.append(np.zeros(3))
            else:
                s = dist / d0
                w = smoothstep(s)
                dw_ds = smoothstep_deriv(s)
                
                # c_eff = (1-w)c_sharp + w*c_orig
                #       = c_sharp + w(c_orig - c_sharp)
                diff = c_orig - c_sharp
                c_val = c_sharp + w * diff
                c_eff.append(c_val)
                
                # Chain rule
                # dc/du = dc/dw * dw/ds * ds/dd * dd/du
                factor = diff * dw_ds * (1.0 / d0)
                dc_du.append(factor * dd_du[i])
                dc_dv.append(factor * dd_dv[i])
    
    # 二次基函数及其导数
    # b1 = (1-u)(u-v)
    b1 = (1.0 - u) * (u - v)
    db1_du = 1.0 - 2.0 * u + v
    db1_dv = u - 1.0
    
    # b2 = (u-v)v
    b2 = (u - v) * v
    db2_du = v
    db2_dv = u - 2.0 * v
    
    # b3 = (1-u)v
    b3 = (1.0 - u) * v
    db3_du = -v
    db3_dv = 1.0 - u
    
    bases = [b1, b2, b3]
    db_du_list = [db1_du, db2_du, db3_du]
    db_dv_list = [db1_dv, db2_dv, db3_dv]
    
    dQ_du = np.zeros(3)
    dQ_dv = np.zeros(3)
    
    for i in range(3):
        # term i: C_i * B_i
        # d/du = dC/du * B + C * dB/du
        term_du = dc_du[i] * bases[i] + c_eff[i] * db_du_list[i]
        term_dv = dc_dv[i] * bases[i] + c_eff[i] * db_dv_list[i]
        
        dQ_du += term_du
        dQ_dv += term_dv
        
    dXdu = dLin_du - dQ_du
    dXdv = dLin_dv - dQ_dv
    
    return dXdu, dXdv

def find_nearest_point_on_patch(
    point: np.ndarray,
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
    is_crease: tuple = (False, False, False),
    c_sharps: tuple = (None, None, None),
    d0: float = 0.1,
    max_iter: int = 15
) -> Tuple[np.ndarray, float, float, float]:
    """
    使用 Newton-Raphson 算法寻找单个 Patch 上的最近点
    (Robust Version: Multiple Restarts)
    Returns: (nearest_point, distance, u, v)
    """
    # 候选初始点列表 (u, v)
    # 包含: 
    # 1. 简单平面投影 (Primary Guess)
    # 2. 面片中心 (重心的Nagata参数大致也在中心附近, 选 u=0.66, v=0.33)
    # 3. 顶点 (0,0), (1,0), (1,1)
    # 4. 边中点 (0.5, 0), (1, 0.5), (0.5, 0.5)
    
    candidates = [
        (0.666, 0.333), # 重心 approximation
        (0.0, 0.0),     # x00
        (1.0, 0.0),     # x10
        (1.0, 1.0),     # x11
        (0.5, 0.0),     # Edge 1 mid
        (1.0, 0.5),     # Edge 2 mid
        (0.5, 0.5)      # Edge 3 mid
    ]

    # 添加平面投影作为候选 (优先级最高)
    edge1 = x10 - x00
    edge2 = x11 - x00
    normal = np.cross(edge1, edge2)
    area_sq = np.dot(normal, normal)
    
    if area_sq > 1e-12:
        w = point - x00
        s = np.dot(np.cross(w, edge2), normal) / area_sq
        t = np.dot(np.cross(edge1, w), normal) / area_sq
        
        # Mapping to u,v assuming linear transform structure approximations
        # u ~ s+t, v ~ t ? 
        # Ideally, we should invert the linear basis x00(1-u) + x10(u-v) + x11v
        # = x00 + u(x10-x00) + v(x11-x10)
        # s corresponds directly to u coeff? t corresponds to v coeff? 
        # No, edge1 = x10-x00, edge2 = x11-x00 (standard)
        # Ours linear: x00 + u*edge1 + v*(x11-x10)
        # x11-x10 = (x11-x00) - (x10-x00) = edge2 - edge1
        # P = x00 + u*edge1 + v*(edge2 - edge1)
        #   = x00 + (u-v)*edge1 + v*edge2
        # So s = u-v, t = v
        # => v = t
        # => u = s + v = s + t
        
        u_proj = np.clip(s + t, 0.0, 1.0)
        v_proj = np.clip(t, 0.0, u_proj)
        candidates.insert(0, (u_proj, v_proj))

    # 挑选最接近的顶点作为参考 (Heuristic)
    d00 = np.sum((point - x00)**2)
    d10 = np.sum((point - x10)**2)
    d11 = np.sum((point - x11)**2)
    if d00 <= d10 and d00 <= d11:
        candidates.insert(1, (0.01, 0.0))
    elif d10 <= d00 and d10 <= d11:
        candidates.insert(1, (0.99, 0.0))
    else:
        candidates.insert(1, (0.99, 0.99))

    # 去重
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = (round(c[0], 2), round(c[1], 2))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    best_dist_sq = float('inf')
    best_res = (x00.copy(), float('inf'), 0.0, 0.0)

    # 对每个候选起点运行优化
    for start_u, start_v in unique_candidates:
        u, v = start_u, start_v
        
        # Optimization Loop
        for _ in range(max_iter):
            # Clamping
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, u)
            
            u_arr, v_arr = np.array([u]), np.array([v])
            
            # Eval
            if any(is_crease):
                P_surf = evaluate_nagata_patch_with_crease(
                    x00, x10, x11, c1, c2, c3, 
                    c_sharps[0], c_sharps[1], c_sharps[2],
                    is_crease, u_arr, v_arr, d0
                ).flatten()
                dXdu, dXdv = evaluate_nagata_derivatives(
                    x00, x10, x11, c1, c2, c3, u, v,
                    is_crease, c_sharps, d0
                )
            else:
                P_surf = evaluate_nagata_patch(
                    x00, x10, x11, c1, c2, c3, u_arr, v_arr
                ).flatten()
                dXdu, dXdv = evaluate_nagata_derivatives(
                    x00, x10, x11, c1, c2, c3, u, v
                )
            
            diff = P_surf - point
            
            # Gradient
            F_u = np.dot(diff, dXdu)
            F_v = np.dot(diff, dXdv)
            
            # Hessian
            H_uu = np.dot(dXdu, dXdu)
            H_uv = np.dot(dXdu, dXdv)
            H_vv = np.dot(dXdv, dXdv)
            
            det = H_uu * H_vv - H_uv * H_uv
            if abs(det) < 1e-9:
                # Hessian invalid, verify gradient descent
                du = -F_u * 0.1 
                dv = -F_v * 0.1
            else:
                inv_det = 1.0 / det
                du = (H_vv * (-F_u) - H_uv * (-F_v)) * inv_det
                dv = (-H_uv * (-F_u) + H_uu * (-F_v)) * inv_det
                
            # Step Limiting
            step_len = np.sqrt(du*du + dv*dv)
            if step_len > 0.3:
                scale = 0.3 / step_len
                du *= scale
                dv *= scale

            u_new = u + du
            v_new = v + dv
            
            # Simple Domain Wall
            if v_new < 0: v_new = 0
            if u_new > 1: u_new = 1
            if u_new < 0: u_new = 0
            if v_new > u_new: v_new = u_new 
                
            change = abs(u_new - u) + abs(v_new - v)
            u, v = u_new, v_new
            
            if change < 1e-6:
                break
                
        # Final Eval for this candidate
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, u)
        
        if any(is_crease):
             P_final = evaluate_nagata_patch_with_crease(
                x00, x10, x11, c1, c2, c3, c_sharps[0], c_sharps[1], c_sharps[2], is_crease, np.array([u]), np.array([v]), d0
            ).flatten()
        else:
            P_final = evaluate_nagata_patch(x00, x10, x11, c1, c2, c3, np.array([u]), np.array([v])).flatten()
            
        dist_sq = np.sum((P_final - point)**2)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_res = (P_final, np.sqrt(dist_sq), u, v)

    return best_res


class NagataModelQuery:
    """
    提供对整个 NSM 模型的最近点查询功能
    (Enhanced: 支持折痕自动检测与 c_sharp 修复)
    """
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, tri_vertex_normals: np.ndarray):
        self.vertices = vertices
        self.triangles = triangles
        self.normals = tri_vertex_normals
        
        # Precompute individual patch data
        self.patch_coeffs = [] # List[Tuple(c1, c2, c3)]
        self.centroids = []
        
        print(f"初始化查询模型: {len(triangles)} 三角形...")
        for i in range(len(triangles)):
            idx = triangles[i]
            x00=vertices[idx[0]]; x10=vertices[idx[1]]; x11=vertices[idx[2]]
            n00=tri_vertex_normals[i,0]; n10=tri_vertex_normals[i,1]; n11=tri_vertex_normals[i,2]
            
            c1, c2, c3 = nagata_patch(x00, x10, x11, n00, n10, n11)
            self.patch_coeffs.append((c1, c2, c3))
            
            # Approximate centroid
            center = (x00 + x10 + x11) / 3.0
            self.centroids.append(center)
            
        self.centroids = np.array(self.centroids)
        
        # Try to build KDTree
        try:
            from scipy.spatial import KDTree
            self.kdtree = KDTree(self.centroids)
            self.use_kdtree = True
            print("KDTree 构建成功，加速开启。")
        except ImportError:
            self.use_kdtree = False
            print("警告: 未找到 scipy.spatial.KDTree，将使用暴力搜索 (速度较慢)。")
            
        # =========================================================
        # 折痕预计算 (Crease Precomputation)
        # =========================================================
        self.crease_map = {} # map edge_key -> c_sharp
        
        # 1. 构建边缘拓扑 edge -> list of (tri_idx, local_edge_idx)
        print("正在构建边缘拓扑以检测折痕...")
        edge_to_tris = {} # (min_v, max_v) -> list of tri_idx
        
        for t_idx in range(len(triangles)):
            # Edge 1: 0-1
            # Edge 2: 1-2
            # Edge 3: 0-2 (注意 Nagata 定义的边序)
            tri = triangles[t_idx]
            edges = [
                tuple(sorted((tri[0], tri[1]))),
                tuple(sorted((tri[1], tri[2]))),
                tuple(sorted((tri[0], tri[2])))
            ]
            
            for e_key in edges:
                if e_key not in edge_to_tris:
                    edge_to_tris[e_key] = []
                edge_to_tris[e_key].append(t_idx)
                
        # 2. 检测折痕并计算 c_sharp
        crease_count = 0
        sharpness_threshold = 0.9999 # cos(angle), > this means smooth
        
        for e_key, tri_indices in edge_to_tris.items():
            if len(tri_indices) != 2:
                continue # 边界边或非流形，暂时跳过修复
                
            t1 = tri_indices[0]
            t2 = tri_indices[1]
            
            # 获取对应的顶点索引和法向量
            def get_edge_data(t_idx, v_a, v_b):
                tri = triangles[t_idx]
                norms = tri_vertex_normals[t_idx]
                
                # Find local indices
                try:
                    idx_a = -1
                    idx_b = -1
                    for k in range(3):
                        if tri[k] == v_a: idx_a = k
                        if tri[k] == v_b: idx_b = k
                    
                    if idx_a == -1 or idx_b == -1: return None, None
                        
                    return norms[idx_a], norms[idx_b]
                except:
                    return None, None

            vA, vB = e_key
            nA_1, nB_1 = get_edge_data(t1, vA, vB)
            nA_2, nB_2 = get_edge_data(t2, vA, vB)
            
            if nA_1 is None or nA_2 is None: continue
            
            # Check consistency
            dot_A = np.dot(nA_1, nA_2)
            dot_B = np.dot(nB_1, nB_2)
            
            # 如果任何一个端点的法向量不一致，视为折痕
            if dot_A < sharpness_threshold or dot_B < sharpness_threshold:
                # 是折痕，计算 c_sharp
                posA = self.vertices[vA]
                posB = self.vertices[vB]
                edge_vec = posB - posA
                
                # 计算切向方向 d_A, d_B
                d_A = compute_crease_direction(nA_1, nA_2, edge_vec)
                d_B = compute_crease_direction(nB_1, nB_2, edge_vec)
                
                # 计算 c_sharp
                c_sharp = compute_c_sharp(posA, posB, d_A, d_B)
                
                self.crease_map[e_key] = c_sharp
                crease_count += 1
                
        if crease_count > 0:
            print(f"检测到 {crease_count} 条折痕边，已启用 c_sharp 修复。")
        else:
            print("未检测到显著折痕，以完全光滑模式运行。")
            
    def _get_patch_crease_info(self, idx: int):
        """Helper to get crease query params for a triangle"""
        tri = self.triangles[idx]
        # Edges order corresponding to c1, c2, c3:
        # 1: v0-v1
        # 2: v1-v2
        # 3: v0-v2
        
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[0], tri[2])))
        ]
        
        is_crease = [False, False, False]
        c_sharps = [None, None, None]
        
        for k in range(3):
            if edges[k] in self.crease_map:
                is_crease[k] = True
                c_sharps[k] = self.crease_map[edges[k]]
                
        return tuple(is_crease), tuple(c_sharps)

    def query(self, point: np.ndarray, k_nearest: int = 16) -> dict:
        """
        查询模型上距离 point 最近的点
        (Enhanced: 支持多解处法向平滑/平均, 支持折痕修复)
        """
        candidate_indices = []
        
        if self.use_kdtree:
            dists, indices = self.kdtree.query(point, k=min(k_nearest, len(self.centroids)))
            if isinstance(indices, (int, np.integer)): indices = [indices]
            candidate_indices = indices
        else:
            candidate_indices = range(len(self.centroids))
            
        # 1. 收集所有候选结果
        candidates = []
        min_dist = float('inf')
        
        for idx in candidate_indices:
            idx = int(idx)
            tri_v_idx = self.triangles[idx]
            x00=self.vertices[tri_v_idx[0]]
            x10=self.vertices[tri_v_idx[1]]
            x11=self.vertices[tri_v_idx[2]]
            c1, c2, c3 = self.patch_coeffs[idx]
            
            # 获取折痕信息
            is_crease, c_sharps = self._get_patch_crease_info(idx)
            
            p_surf, dist, u, v = find_nearest_point_on_patch(
                point, x00, x10, x11, c1, c2, c3,
                is_crease=is_crease, c_sharps=c_sharps
            )
            
            if dist < min_dist:
                min_dist = dist
            
            # 计算法向备用
            dXdu, dXdv = evaluate_nagata_derivatives(
                x00, x10, x11, c1, c2, c3, u, v,
                is_crease=is_crease, c_sharps=c_sharps
            )
            normal = np.cross(dXdu, dXdv)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal /= norm_len
            else:
                normal = np.array([0.,0.,1.])
                
            candidates.append({
                'nearest_point': p_surf,
                'distance': dist,
                'normal': normal,
                'triangle_index': idx,
                'uv': (u,v)
            })
            
        # 2. 筛选最优集合 (Tolerance for float errors)
        epsilon = 1e-5 * (min_dist + 1.0) # Relative + Absolute
        best_candidates = [c for c in candidates if c['distance'] <= min_dist + epsilon]
        
        if not best_candidates:
            return None
            
        # 3. 融合结果 (Selector implementation)
        # 核心逻辑: 平均所有"最近"候选者的梯度方向
        # - 对于外部 Corner (Same point): geometry direction 是一致的 (P-q)，平均无影响 (除了数值噪点)
        # - 对于内部 Medial Axis (Different points): geometry directions 不同，平均产生对称的角平分线 (Theory Section 5.2)
        
        avg_gradient = np.zeros(3)
        mean_dist = 0.0
        contributing_count = 0
        
        # 记录用于显示的元数据 (取第一个)
        primary_res = best_candidates[0]
        
        for c in best_candidates:
            p_surf = c['nearest_point']
            surf_normal = c['normal']
            dist = c['distance']
            
            # 计算该候选者的梯度方向 g_i
            diff_vec = point - p_surf
            dist_geo = np.linalg.norm(diff_vec)
            
            if dist_geo > 1e-6:
                # 几何方向 (P - q) / d
                g_i = diff_vec / dist_geo
                
                # 符号判断: SDF Gradient 永远指向"SDF值增加"的方向 (Outwards)
                # Check alignment with surface normal
                if np.dot(g_i, surf_normal) < 0:
                     # P is Inside. P-q points "Inwards".
                     # Gradient should point "Outwards" (-g_i).
                     g_i = -g_i
                else:
                     # P is Outside. P-q points "Outwards".
                     # Gradient = g_i
                     pass
            else:
                # On surface: use surface normal
                g_i = surf_normal
            
            avg_gradient += g_i
            mean_dist += dist
            contributing_count += 1
            
        if contributing_count > 0:
            # 归一化平均梯度
            norm_len = np.linalg.norm(avg_gradient)
            if norm_len > 1e-12:
                avg_gradient /= norm_len
            else:
                # Fallback (Theory Section 5.3): Maximum Inner Product or Parent
                # 这里简单处理: 取第一个的梯度
                avg_gradient = primary_res['normal'] # This is actually surf normal in raw data... 
                # Recompute exact gradient for primary
                diff = point - primary_res['nearest_point']
                if np.linalg.norm(diff) > 1e-6:
                    g0 = diff / np.linalg.norm(diff)
                    if np.dot(g0, primary_res['normal']) < 0: g0 = -g0
                    avg_gradient = g0
                
            mean_dist /= contributing_count
            
            # 更新结果
            primary_res['normal'] = avg_gradient
            primary_res['distance'] = mean_dist
            
            # Signed Distance (Re-evaluate sign based on final averaged gradient?)
            # Usually sign is determined by the "Winner".
            # If internal, sign is negative.
            # But the 'query' function returns unsigned distance + normal.
            # The calling script calculates signed distance.
            # We should ensure consistence. Use dot(P-q, N_final)?
            # No, `query` returns 'distance' (unsigned).
            # Caller does `sign = 1 if dot(diff, normal) >= 0 else -1`.
            # If we return N_final = (1,1).normalized(). P=(0.4,0.4). q=(0.5,0.4)?
            # diff = (-0.1, 0). dot((-0.1, 0), (0.7, 0.7)) = -0.07 < 0.
            # Sign = -1. Correct.
            
        return primary_res


if __name__ == '__main__':
    # 简单测试
    print("Nagata Patch 计算模块测试")
    
    # 测试一个简单三角形
    x00 = np.array([0.0, 0.0, 0.0])
    x10 = np.array([1.0, 0.0, 0.0])
    x11 = np.array([0.5, 1.0, 0.0])
    
    # 假设法向量都指向z轴正方向
    n00 = np.array([0.0, 0.0, 1.0])
    n10 = np.array([0.0, 0.0, 1.0])
    n11 = np.array([0.0, 0.0, 1.0])
    
    c1, c2, c3 = nagata_patch(x00, x10, x11, n00, n10, n11)
    print(f"曲率系数: c1={c1}, c2={c2}, c3={c3}")
    
    # 采样测试
    verts, faces = sample_nagata_triangle(x00, x10, x11, n00, n10, n11, resolution=5)
    print(f"采样结果: {len(verts)} 个顶点, {len(faces)} 个三角形")

    # 投影测试
    print("\n投影测试:")
    model = NagataModelQuery(
        vertices=np.array([x00, x10, x11]),
        triangles=np.array([[0, 1, 2]]),
        tri_vertex_normals=np.array([[n00, n10, n11]])
    )
    
    test_pt = np.array([0.5, 0.3, 1.0]) # 位于三角形上方
    res = model.query(test_pt)
    print(f"查询点: {test_pt}")
    print(f"最近点: {res['nearest_point']}")
    print(f"距离: {res['distance']:.6f}")
    print(f"法向: {res['normal']}")
    print(f"Face ID: {res['triangle_index']}")

