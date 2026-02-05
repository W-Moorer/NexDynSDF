"""
OBJ文件Nagata可视化工具
支持无法向量OBJ文件，使用加权平均计算顶点法向量

四种权重方案（来自Nagata论文）：
1. MWE (Mean Weighted Equally): 简单平均
2. MWA (Mean Weighted by Angle): 按邻接角度加权
3. MWAAT (Mean Weighted by Areas of Adjacent Triangles): 按面积加权
4. MWSELR (Mean Weighted by Sine and Edge Length Reciprocals): 适用于球面

依赖：
- numpy
- pyvista
- trimesh
"""

import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path
from typing import Tuple, Optional, Literal
from enum import Enum
import sys

# 导入本地模块
from nagata_patch import sample_all_nagata_patches


class NormalWeightingScheme(Enum):
    """法向量权重方案"""
    MWE = "mwe"      # Mean Weighted Equally (简单平均)
    MWA = "mwa"      # Mean Weighted by Angle (角度加权)
    MWAAT = "mwaat"  # Mean Weighted by Areas (面积加权)
    MWSELR = "mwselr"  # Mean Weighted by Sine and Edge Length Reciprocals


def compute_vertex_normals_weighted(
    vertices: np.ndarray,
    faces: np.ndarray,
    scheme: NormalWeightingScheme = NormalWeightingScheme.MWSELR
) -> np.ndarray:
    """
    计算顶点法向量（加权平均方案）
    
    基于Nagata论文的四种权重方案计算顶点法向量。
    
    Args:
        vertices: 顶点坐标, shape (num_vertices, 3)
        faces: 三角形索引, shape (num_faces, 3)
        scheme: 权重方案
        
    Returns:
        vertex_normals: 顶点法向量, shape (num_vertices, 3)
    """
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # 初始化顶点法向量累加器
    vertex_normals = np.zeros((num_vertices, 3), dtype=np.float64)
    
    for face_idx in range(num_faces):
        i0, i1, i2 = faces[face_idx]
        
        # 顶点坐标
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        
        # 边向量
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2
        
        # 面法向量 (未归一化)
        face_normal = np.cross(e01, -e20)
        face_normal_norm = np.linalg.norm(face_normal)
        
        if face_normal_norm < 1e-12:
            continue  # 退化三角形
        
        # 单位面法向量
        face_normal_unit = face_normal / face_normal_norm
        
        # 计算各顶点的权重
        if scheme == NormalWeightingScheme.MWE:
            # 简单平均，权重为1
            w0 = w1 = w2 = 1.0
            
        elif scheme == NormalWeightingScheme.MWA:
            # 按角度加权
            # 计算各顶点处的角度
            def angle_between(v1, v2):
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                return np.arccos(np.clip(cos_angle, -1, 1))
            
            w0 = angle_between(e01, -e20)
            w1 = angle_between(-e01, e12)
            w2 = angle_between(-e12, e20)
            
        elif scheme == NormalWeightingScheme.MWAAT:
            # 按面积加权 (对所有顶点使用相同权重=面积)
            area = face_normal_norm * 0.5
            w0 = w1 = w2 = area
            
        elif scheme == NormalWeightingScheme.MWSELR:
            # 按 sin(angle) / (||e1|| * ||e2||) 加权
            # 这是最精确的方案，对球面可以给出精确法向
            def compute_mwselr_weight(e1, e2):
                len_e1 = np.linalg.norm(e1)
                len_e2 = np.linalg.norm(e2)
                if len_e1 < 1e-12 or len_e2 < 1e-12:
                    return 0.0
                # sin(angle) = ||e1 x e2|| / (||e1|| * ||e2||)
                cross = np.cross(e1, e2)
                sin_angle = np.linalg.norm(cross) / (len_e1 * len_e2)
                return sin_angle / (len_e1 * len_e2)
            
            w0 = compute_mwselr_weight(e01, -e20)
            w1 = compute_mwselr_weight(-e01, e12)
            w2 = compute_mwselr_weight(-e12, e20)
        else:
            w0 = w1 = w2 = 1.0
        
        # 累加加权法向量
        vertex_normals[i0] += w0 * face_normal_unit
        vertex_normals[i1] += w1 * face_normal_unit
        vertex_normals[i2] += w2 * face_normal_unit
    
    # 归一化
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # 避免除零
    vertex_normals = vertex_normals / norms
    
    return vertex_normals


def compute_per_triangle_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray
) -> np.ndarray:
    """
    从顶点法向量生成每个三角形的顶点法向量数组
    
    Args:
        vertices: 顶点坐标, shape (num_vertices, 3)
        faces: 三角形索引, shape (num_faces, 3)
        vertex_normals: 顶点法向量, shape (num_vertices, 3)
        
    Returns:
        tri_vertex_normals: 每个三角形的顶点法向量, shape (num_faces, 3, 3)
    """
    num_faces = len(faces)
    tri_vertex_normals = np.zeros((num_faces, 3, 3), dtype=np.float64)
    
    for face_idx in range(num_faces):
        i0, i1, i2 = faces[face_idx]
        tri_vertex_normals[face_idx, 0] = vertex_normals[i0]
        tri_vertex_normals[face_idx, 1] = vertex_normals[i1]
        tri_vertex_normals[face_idx, 2] = vertex_normals[i2]
    
    return tri_vertex_normals


def force_merge_vertices(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    按位置强制合并顶点
    
    Args:
        vertices: 顶点坐标
        faces: 面索引
        
    Returns:
        new_vertices: 合并后的顶点
        new_faces: 更新后的面索引
    """
    # 使用numpy.unique按行(坐标)查找唯一顶点
    # round以避免浮点误差
    vertices_rounded = np.round(vertices, decimals=6)
    _, unique_indices, inverse_indices = np.unique(
        vertices_rounded, axis=0, return_index=True, return_inverse=True
    )
    
    new_vertices = vertices[unique_indices]
    new_faces = inverse_indices[faces]
    
    return new_vertices, new_faces


def load_obj_mesh(
    filepath: str, 
    merge_vertices: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    加载OBJ文件
    
    Args:
        filepath: OBJ文件路径
        merge_vertices: 是否强制按位置合并顶点
        
    Returns:
        vertices: 顶点坐标
        faces: 三角形索引
        vertex_normals: 顶点法向量 (如果有)，否则返回None
    """
    mesh = trimesh.load(filepath, force='mesh')
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    # trimesh自动计算顶点法向量
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        vertex_normals = np.asarray(mesh.vertex_normals)
    else:
        vertex_normals = None
    
    print(f"OBJ文件信息:")
    print(f"  原始顶点数: {len(vertices)}")
    print(f"  原始三角形数: {len(faces)}")
    
    if merge_vertices:
        print("执行强制顶点合并 (按位置)...")
        vertices, faces = force_merge_vertices(vertices, faces)
        print(f"  合并后顶点数: {len(vertices)}")
        # 合并顶点后，原有的法向量不再对应，必须丢弃
        vertex_normals = None
        
    print(f"  法向量: {'已有' if vertex_normals is not None else '需计算'}")
    
    return vertices, faces, vertex_normals


def create_nagata_mesh_from_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    tri_vertex_normals: np.ndarray,
    resolution: int = 10
) -> pv.PolyData:
    """
    从OBJ数据创建Nagata曲面的PyVista网格
    """
    # 采样所有Nagata patches
    nagata_verts, nagata_faces, face_to_original = sample_all_nagata_patches(
        vertices, faces, tri_vertex_normals, resolution
    )
    
    if len(nagata_verts) == 0:
        print("警告: 无法生成Nagata曲面")
        return pv.PolyData()
    
    # 创建PyVista格式的faces数组
    pv_faces = np.hstack([
        np.full((nagata_faces.shape[0], 1), 3, dtype=np.int32),
        nagata_faces.astype(np.int32)
    ]).flatten()
    
    # 创建PolyData
    mesh = pv.PolyData(nagata_verts, pv_faces)
    
    if len(face_to_original) > 0:
        mesh.cell_data['original_tri'] = face_to_original
    
    return mesh


def visualize_obj_nagata(
    filepath: str,
    resolution: int = 10,
    weighting_scheme: str = "mwselr",
    recompute_normals: bool = False,
    merge_vertices: bool = False,
    show_comparison: bool = True,
    show_edges: bool = False
):
    """
    可视化OBJ文件的Nagata曲面
    
    Args:
        filepath: OBJ文件路径
        resolution: Nagata采样密度
        weighting_scheme: 法向量权重方案 (mwe/mwa/mwaat/mwselr)
        recompute_normals: 强制重新计算法向量
        show_comparison: 是否显示原始网格对比
        show_edges: 是否显示网格边
    """
    # 加载OBJ文件
    print(f"加载文件: {filepath}")
    vertices, faces, vertex_normals_from_file = load_obj_mesh(filepath, merge_vertices=merge_vertices)
    
    # 确定是否需要计算法向量
    if recompute_normals or vertex_normals_from_file is None:
        # 解析权重方案
        scheme_map = {
            'mwe': NormalWeightingScheme.MWE,
            'mwa': NormalWeightingScheme.MWA,
            'mwaat': NormalWeightingScheme.MWAAT,
            'mwselr': NormalWeightingScheme.MWSELR
        }
        scheme = scheme_map.get(weighting_scheme.lower(), NormalWeightingScheme.MWSELR)
        
        print(f"计算顶点法向量 (方案: {scheme.value})...")
        vertex_normals = compute_vertex_normals_weighted(vertices, faces, scheme)
    else:
        print("使用文件中的法向量...")
        vertex_normals = vertex_normals_from_file
    
    # 生成每个三角形的顶点法向量
    tri_vertex_normals = compute_per_triangle_vertex_normals(vertices, faces, vertex_normals)
    
    print(f"计算Nagata曲面 (分辨率={resolution})...")
    
    # 创建原始网格
    pv_faces = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ]).flatten()
    original_mesh = pv.PolyData(vertices, pv_faces)
    
    # 创建Nagata曲面网格
    nagata_mesh = create_nagata_mesh_from_obj(
        vertices, faces, tri_vertex_normals, resolution
    )
    
    print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
    
    if show_comparison:
        # 并排对比显示
        plotter = pv.Plotter(shape=(1, 2))
        
        # 左边: 原始网格
        plotter.subplot(0, 0)
        plotter.add_text("原始网格 (OBJ)", font_size=12, position='upper_edge')
        plotter.set_background('white')
        plotter.add_mesh(
            original_mesh,
            color='lightblue',
            show_edges=show_edges,
            opacity=1.0
        )
        plotter.add_axes()
        
        # 右边: Nagata曲面
        plotter.subplot(0, 1)
        plotter.add_text(f"Nagata曲面 ({weighting_scheme.upper()})", 
                        font_size=12, position='upper_edge')
        plotter.set_background('white')
        plotter.add_mesh(
            nagata_mesh,
            color='lightblue',
            show_edges=show_edges,
            opacity=1.0
        )
        plotter.add_axes()
        
        # 链接两个视图的相机
        plotter.link_views()
        
    else:
        # 单独显示Nagata曲面
        plotter = pv.Plotter()
        plotter.set_background('white')
        plotter.add_mesh(
            nagata_mesh,
            color='lightblue',
            show_edges=show_edges,
            opacity=1.0
        )
        plotter.add_axes()
        plotter.add_title(
            f"Nagata曲面 (分辨率={resolution}, 方案={weighting_scheme.upper()})",
            font_size=12
        )
    
    print("\n交互式可视化已启动:")
    print("  - 左键拖动: 旋转")
    print("  - 右键拖动: 缩放")
    print("  - 中键拖动: 平移")
    print("  - 滚轮: 缩放")
    print("  - 'q': 退出")
    
    plotter.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OBJ文件Nagata可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
法向量权重方案:
  mwe    - Mean Weighted Equally (简单平均)
  mwa    - Mean Weighted by Angle (角度加权)
  mwaat  - Mean Weighted by Areas (面积加权)
  mwselr - Mean Weighted by Sine and Edge Length Reciprocals (推荐)

示例:
  python visualize_obj_nagata.py ../models/obj/complex_geometry/Gear.obj
  python visualize_obj_nagata.py model.obj -r 15 --scheme mwselr
  python visualize_obj_nagata.py model.obj -r 15 --scheme mwselr
  python visualize_obj_nagata.py model.obj --recompute-normals --scheme mwa
  python visualize_obj_nagata.py model.obj --merge-vertices --scheme mwselr  # 修复破面
        """
    )
    
    parser.add_argument('filepath', help='OBJ文件路径')
    parser.add_argument('-r', '--resolution', type=int, default=10,
                        help='Nagata采样密度 (默认: 10)')
    parser.add_argument('--scheme', choices=['mwe', 'mwa', 'mwaat', 'mwselr'],
                        default='mwselr', help='法向量权重方案 (默认: mwselr)')
    parser.add_argument('--recompute-normals', action='store_true',
                        help='强制重新计算法向量')
    parser.add_argument('--merge-vertices', action='store_true',
                        help='强制合并空间重合的顶点 (修复破面)')
    parser.add_argument('--no-compare', action='store_true',
                        help='不显示原始网格对比')
    parser.add_argument('--edges', action='store_true',
                        help='显示网格边')
    
    args = parser.parse_args()
    
    visualize_obj_nagata(
        args.filepath,
        resolution=args.resolution,
        weighting_scheme=args.scheme,
        recompute_normals=args.recompute_normals,
        merge_vertices=args.merge_vertices,
        show_comparison=not args.no_compare,
        show_edges=args.edges
    )


if __name__ == '__main__':
    main()
