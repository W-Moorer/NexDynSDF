"""
Nagata Patch 裂缝检测工具

该工具用于检测 NSM 模型中是否存在"裂缝"隐患。
Nagata Patch 插值依赖于连续的法向量场。如果同一个顶点在不同面片上
具有不一致的法向量（Shared Vertex, Divergent Normals），
会导致生成的 Nagata 曲面在该顶点处不重合，从而产生可见的裂缝。

检测原理：
1. 遍历所有唯一顶点。
2. 收集该顶点在所有关联三角形中的法向量。
3. 计算法向量的差异（最大夹角）。
4. 如果差异超过阈值，标记为"裂缝顶点"。
"""

import numpy as np
import pyvista as pv
import sys
import argparse
from pathlib import Path

# 尝试导入本地模块
try:
    from nsm_reader import load_nsm, create_pyvista_mesh
    from nagata_patch import compute_curvature
except ImportError:
    # 如果在上一级目录运行，可能需要调整路径
    sys.path.append(str(Path(__file__).parent))
    from nsm_reader import load_nsm, create_pyvista_mesh
    from nagata_patch import compute_curvature


def calculate_max_angle(normals: np.ndarray) -> float:
    """计算一组法向量中的最大夹角（度）"""
    if len(normals) < 2:
        return 0.0
    
    # 归一化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms.flatten() > 1e-12
    if not np.all(valid):
        normals = normals[valid]
        norms = norms[valid]
        if len(normals) < 2:
            return 0.0
            
    normals = normals / norms
    
    # 计算两两最大夹角
    max_angle_rad = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            dot = np.dot(normals[i], normals[j])
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            if angle > max_angle_rad:
                max_angle_rad = angle
                
    return np.degrees(max_angle_rad)


def evaluate_edge_curve_midpoint(x0: np.ndarray, x1: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """计算 Nagata 边曲线的中点 (t=0.5)"""
    # 计算曲率系数 c
    # 注意: nagata_patch.compute_curvature 需要 (x1-x0), n0, n1
    d = x1 - x0
    c = compute_curvature(d, n0, n1)
    
    # Nagata 曲线方程 (Edge):
    # x(t) = (1-t)x0 + t*x1 - c*t*(1-t)
    # At t=0.5:
    # x(0.5) = 0.5*x0 + 0.5*x1 - 0.25*c
    
    midpoint = 0.5 * (x0 + x1) - 0.25 * c
    return midpoint


def check_cracks(
    filepath: str,
    gap_threshold: float = 1e-4,  # 几何缝隙阈值 (例如 0.0001 单位)
    angle_threshold: float = 1.0,  # 角度阈值 (仅用于辅助显示的顶点分析)
    visualize: bool = True
):
    """
    执行检测并可视化 (基于几何缝隙计算)
    """
    print(f"正在分析文件: {filepath}")
    mesh_data = load_nsm(filepath)
    
    num_vertices = mesh_data.vertices.shape[0]
    num_triangles = mesh_data.triangles.shape[0]
    
    print(f"网格统计: {num_vertices} 顶点, {num_triangles} 三角形")
    
    # --- Part 1: 构建拓扑 (Edge -> Triangles) ---
    print("构建边缘拓扑...")
    edge_to_tris = {} # key: (min_v, max_v), value: list of tri_idx
    
    triangles = mesh_data.triangles
    
    for t_idx in range(num_triangles):
        for k in range(3):
            v1 = triangles[t_idx, k]
            v2 = triangles[t_idx, (k+1)%3]
            
            edge_key = tuple(sorted((v1, v2)))
            if edge_key not in edge_to_tris:
                edge_to_tris[edge_key] = []
            edge_to_tris[edge_key].append(t_idx)
            
    # --- Part 1.5: 收集并检测顶点法向量不一致 (Vertex Analysis) ---
    print("正在检查顶点法向量一致性...")
    vertex_normals = [[] for _ in range(num_vertices)]
    normals = mesh_data.tri_vertex_normals
    for t_idx in range(num_triangles):
        for k in range(3):
            v_idx = triangles[t_idx, k]
            n = normals[t_idx, k]
            vertex_normals[v_idx].append(n)
            
    cracked_vertices = []
    max_angles = []
    
    for v_idx in range(num_vertices):
        v_norms = vertex_normals[v_idx]
        if len(v_norms) < 2:
            continue
        angle_deg = calculate_max_angle(np.array(v_norms))
        if angle_deg > angle_threshold:
            cracked_vertices.append(v_idx)
            max_angles.append(angle_deg)
            
    print(f"发现 {len(cracked_vertices)} 个法向量不一致顶点 (> {angle_threshold} deg)")

    # --- Part 2: 计算几何缝隙 (Gap Analysis) ---
    print(f"正在分析 {len(edge_to_tris)} 条边的几何缝隙...")
    
    cracked_edges_info = [] # (v1, v2, gap_size, max_angle_diff)
    max_gap_found = 0.0
    
    vertices = mesh_data.vertices
    tri_normals = mesh_data.tri_vertex_normals
    
    for edge_key, tri_indices in edge_to_tris.items():
        if len(tri_indices) < 2:
            continue # 边界边，没有"另一侧"，无法产生内部缝隙
            
        v_idx_a, v_idx_b = edge_key
        pos_a = vertices[v_idx_a]
        pos_b = vertices[v_idx_b]
        
        # 为了简化，我们只比较前两个相邻的三角形 (对于非Manifold网格可能需要根据 winding order 比较)
        # 绝大多数情况下 len(tri_indices) == 2
        
        # 获取第一个三角形关于此边的数据
        t1 = tri_indices[0]
        # 找到 v_a, v_b 在 t1 中的局部索引
        # 注意: nsm_reader 中 tri_vertex_normals 顺序对应 triangles 顶点顺序
        # 我们需要准确匹配顶点位置对应的法向量
        
        t1_verts = triangles[t1]
        try:
            # np.where 返回 tuple(array), 取 [0][0]
            local_idx_a_t1 = np.where(t1_verts == v_idx_a)[0][0]
            local_idx_b_t1 = np.where(t1_verts == v_idx_b)[0][0]
        except IndexError:
            continue # Should not happen
            
        n_a_t1 = tri_normals[t1, local_idx_a_t1]
        n_b_t1 = tri_normals[t1, local_idx_b_t1]
        
        # 计算面片1推导的边中点
        # 注意方向: x0->x1 vs x1->x0 对 c 的影响?
        # magnitude of c is distinct, formula handles vector subtraction. 
        # But we must be consistent. Let's use v_idx_a -> v_idx_b as reference direction.
        mid1 = evaluate_edge_curve_midpoint(pos_a, pos_b, n_a_t1, n_b_t1)
        
        # 与其他共享此边的三角形比较
        for i in range(1, len(tri_indices)):
            t2 = tri_indices[i]
            t2_verts = triangles[t2]
            try:
                local_idx_a_t2 = np.where(t2_verts == v_idx_a)[0][0]
                local_idx_b_t2 = np.where(t2_verts == v_idx_b)[0][0]
            except IndexError:
                continue
                
            n_a_t2 = tri_normals[t2, local_idx_a_t2]
            n_b_t2 = tri_normals[t2, local_idx_b_t2]
            
            mid2 = evaluate_edge_curve_midpoint(pos_a, pos_b, n_a_t2, n_b_t2)
            
            # 计算距离 (Gap Size)
            gap = np.linalg.norm(mid1 - mid2)
            
            if gap > gap_threshold:
                # 记录裂缝
                # 顺便计算一下法向量夹角差异用于参考
                # 计算 (na1 vs na2) 和 (nb1 vs nb2) 的最大角
                dot_a = np.clip(np.dot(n_a_t1, n_a_t2), -1, 1)
                dot_b = np.clip(np.dot(n_b_t1, n_b_t2), -1, 1)
                angle_diff = max(np.degrees(np.arccos(dot_a)), np.degrees(np.arccos(dot_b)))
                
                cracked_edges_info.append({
                    'v1': v_idx_a,
                    'v2': v_idx_b,
                    'gap': gap,
                    'angle': angle_diff
                })
                max_gap_found = max(max_gap_found, gap)
                # 只要发现一个不一致就可以标记这条边了
                break
                
    num_cracked_edges = len(cracked_edges_info)
    
    print("=" * 50)
    print(f"几何检测结果 (Gap 阈值={gap_threshold}):")
    if num_cracked_edges == 0:
        print("未发现明显的几何裂缝。")
    else:
        print(f"发现 {num_cracked_edges} 条存在裂缝的边！")
        print(f"最大缝隙宽度: {max_gap_found:.6f}")
        avg_gap = np.mean([e['gap'] for e in cracked_edges_info])
        print(f"平均缝隙宽度: {avg_gap:.6f}")
    print("=" * 50)

    # --- Part 3: 可视化 ---
    if visualize and (num_cracked_edges > 0 or len(cracked_vertices) > 0):
        print("启动可视化...")
        plotter = pv.Plotter()
        plotter.set_background('white')
        
        # 1. 基础网格 (线框)
        grid = create_pyvista_mesh(mesh_data)
        plotter.add_mesh(grid, color='lightblue', opacity=0.2, show_edges=True, label='Base Mesh')
        
        # 1.5 高亮裂缝顶点 (恢复此功能)
        if len(cracked_vertices) > 0:
            crack_points = mesh_data.vertices[cracked_vertices]
            plotter.add_mesh(
                crack_points, 
                scalars=np.array(max_angles),
                render_points_as_spheres=True,
                point_size=10,
                cmap='jet',
                label='Inconsistent Vertices',
                show_scalar_bar=False
            )

        # 2. 高亮裂缝边
        if num_cracked_edges > 0:
            # 构建 PolyData
            lines_list = []
            gap_scalars = []
            
            for info in cracked_edges_info:
                lines_list.extend([2, info['v1'], info['v2']])
                gap_scalars.append(info['gap'])
            
            lines_array = np.array(lines_list)
            crack_poly = pv.PolyData(vertices, lines=lines_array)
            crack_poly.cell_data['Gap Size'] = gap_scalars
            
            plotter.add_mesh(
                crack_poly,
                scalars='Gap Size',
                line_width=5,
                cmap='plasma',
                render_lines_as_tubes=True, # 让线更有体积感
                label='Cracked Edges',
                show_scalar_bar=False
            )
        
        plotter.add_text(
            f"Detected {num_cracked_edges} Cracked Edges\n"
            f"Detected {len(cracked_vertices)} Inconsistent Vertices\n"
            f"Max Gap: {max_gap_found:.5f}", 
            position='upper_left',
            font_size=12
        )
        
        plotter.add_legend()
        plotter.show()
    elif visualize:
        print("无裂缝可显示。")
        
    return cracked_edges_info


def main():
    parser = argparse.ArgumentParser(description='检测 Nagata Patch 几何裂缝 (基于曲线计算)')
    parser.add_argument('filepath', help='NSM文件路径')
    parser.add_argument('-g', '--gap', type=float, default=1e-4, 
                        help='几何缝隙阈值(Distance). 默认=1e-4')
    parser.add_argument('--no-vis', action='store_true', help='不显示可视化窗口')
    
    args = parser.parse_args()
    
    check_cracks(args.filepath, gap_threshold=args.gap, visualize=not args.no_vis)


if __name__ == '__main__':
    main()
