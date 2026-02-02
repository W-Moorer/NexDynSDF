"""
Nagata Patch可视化工具
使用PyVista进行交互式可视化

功能:
- 读取NSM文件
- 计算Nagata曲面
- 并排对比原始网格与Nagata曲面
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import sys

# 导入本地模块
from nsm_reader import load_nsm, create_pyvista_mesh
from nagata_patch import sample_all_nagata_patches


def create_nagata_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    tri_face_ids: np.ndarray,
    resolution: int = 10
) -> pv.PolyData:
    """
    从NSM数据创建Nagata曲面的PyVista网格
    
    Args:
        vertices: 顶点坐标
        triangles: 三角形索引
        tri_vertex_normals: 顶点法向量
        tri_face_ids: 面片ID
        resolution: 采样密度
        
    Returns:
        pv.PolyData: Nagata曲面网格
    """
    # 采样所有Nagata patches
    nagata_verts, nagata_faces, face_to_original = sample_all_nagata_patches(
        vertices, triangles, tri_vertex_normals, resolution
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
    
    # 添加面片ID (从原始三角形继承)
    if len(face_to_original) > 0:
        mesh.cell_data['face_id'] = tri_face_ids[face_to_original]
        mesh.cell_data['original_tri'] = face_to_original
    
    return mesh


def visualize_nagata(
    filepath: str,
    resolution: int = 10,
    show_comparison: bool = True,
    show_edges: bool = False,
    color_by_face_id: bool = False
):
    """
    可视化NSM文件的Nagata曲面
    
    Args:
        filepath: NSM文件路径
        resolution: Nagata采样密度
        show_comparison: 是否显示原始网格对比
        show_edges: 是否显示网格边
        color_by_face_id: 是否按面片ID着色
    """
    # 加载NSM数据
    print(f"加载文件: {filepath}")
    mesh_data = load_nsm(filepath)
    
    print(f"计算Nagata曲面 (分辨率={resolution})...")
    
    # 创建原始网格
    original_mesh = create_pyvista_mesh(mesh_data)
    
    # 创建Nagata曲面网格
    nagata_mesh = create_nagata_mesh(
        mesh_data.vertices,
        mesh_data.triangles,
        mesh_data.tri_vertex_normals,
        mesh_data.tri_face_ids,
        resolution
    )
    
    print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
    
    # 设置颜色方案
    if color_by_face_id:
        scalars = 'face_id'
        cmap = 'tab20'
    else:
        scalars = None
        cmap = None
    
    if show_comparison:
        # 并排对比显示
        plotter = pv.Plotter(shape=(1, 2))
        
        # 左边: 原始网格
        plotter.subplot(0, 0)
        plotter.add_text("原始网格", font_size=12, position='upper_edge')
        plotter.set_background('white')
        
        if color_by_face_id:
            plotter.add_mesh(
                original_mesh,
                scalars=scalars,
                cmap=cmap,
                show_edges=show_edges,
                opacity=1.0
            )
        else:
            plotter.add_mesh(
                original_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=1.0
            )
        plotter.add_axes()
        
        # 右边: Nagata曲面
        plotter.subplot(0, 1)
        plotter.add_text("Nagata曲面", font_size=12, position='upper_edge')
        plotter.set_background('white')
        
        if color_by_face_id and 'face_id' in nagata_mesh.cell_data:
            plotter.add_mesh(
                nagata_mesh,
                scalars='face_id',
                cmap=cmap,
                show_edges=show_edges,
                opacity=1.0
            )
        else:
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
        
        if color_by_face_id and 'face_id' in nagata_mesh.cell_data:
            plotter.add_mesh(
                nagata_mesh,
                scalars='face_id',
                cmap=cmap,
                show_edges=show_edges,
                opacity=1.0
            )
            plotter.add_scalar_bar(title='Face ID')
        else:
            plotter.add_mesh(
                nagata_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=1.0
            )
        
        plotter.add_axes()
        plotter.add_title(
            f"Nagata曲面 (分辨率={resolution})",
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
        description='Nagata Patch可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_nagata.py ../models/nsm/Gear_I.nsm
  python visualize_nagata.py ../models/nsm/Gear_I.nsm -r 20 --edges
  python visualize_nagata.py ../models/nsm/Gear_I.nsm --no-compare --color-by-id
        """
    )
    
    parser.add_argument('filepath', help='NSM文件路径')
    parser.add_argument('-r', '--resolution', type=int, default=10,
                        help='Nagata采样密度 (默认: 10)')
    parser.add_argument('--no-compare', action='store_true',
                        help='不显示原始网格对比')
    parser.add_argument('--edges', action='store_true',
                        help='显示网格边')
    parser.add_argument('--color-by-id', action='store_true',
                        help='按面片ID着色')
    
    args = parser.parse_args()
    
    visualize_nagata(
        args.filepath,
        resolution=args.resolution,
        show_comparison=not args.no_compare,
        show_edges=args.edges,
        color_by_face_id=args.color_by_id
    )


if __name__ == '__main__':
    main()
