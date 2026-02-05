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
