#!/usr/bin/env python3
"""
增强Nagata曲面HTML可视化工具
使用Plotly生成交互式HTML文件，支持并排对比Mesh、Nagata、Enhanced三种模式

使用方法:
    python visualize_nagata_html.py <nsm_file> [options]

示例:
    # 基础用法
    python visualize_nagata_html.py model.nsm
    
    # 启用增强模式
    python visualize_nagata_html.py model.nsm --enhance
    
    # 调整采样分辨率
    python visualize_nagata_html.py model.nsm --enhance --resolution 20
    
    # 按面片ID着色
    python visualize_nagata_html.py model.nsm --enhance --color-by-face-id
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple

from nsm_reader import load_nsm, create_pyvista_mesh
from nagata_patch import sample_all_nagata_patches
from nagata_storage import get_eng_filepath, has_cached_data, load_enhanced_data
from visualize_nagata import (
    create_nagata_mesh, create_nagata_mesh_enhanced,
    compute_average_normals
)


def polydata_to_arrays(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    将PyVista PolyData转换为顶点和面片数组
    
    Args:
        mesh: PyVista PolyData对象
        
    Returns:
        vertices: 顶点数组 (N, 3)
        faces: 面片数组 (M, 3)
    """
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    
    vertices = mesh.points
    
    faces = mesh.faces
    if faces.size == 0:
        return vertices, np.zeros((0, 3), dtype=int)
    
    face_indices = []
    i = 0
    while i < len(faces):
        n_verts = faces[i]
        if n_verts == 3:
            face_indices.append([faces[i+1], faces[i+2], faces[i+3]])
        i += n_verts + 1
    
    return vertices, np.array(face_indices, dtype=int)


def create_mesh_trace(vertices: np.ndarray, faces: np.ndarray, 
                      name: str = "Mesh", color: str = "lightblue",
                      opacity: float = 1.0, show_edges: bool = False,
                      face_colors: Optional[np.ndarray] = None,
                      colorbar_title: str = ""):
    """
    创建Plotly Mesh3d trace
    
    Args:
        vertices: 顶点数组 (N, 3)
        faces: 面片数组 (M, 3)
        name: 图例名称
        color: 基础颜色
        opacity: 透明度
        show_edges: 是否显示边线
        face_colors: 面片颜色数组 (M,)
        colorbar_title: 颜色条标题
        
    Returns:
        Plotly Mesh3d对象
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Error: 需要安装plotly库")
        print("  pip install plotly")
        sys.exit(1)
    
    if len(vertices) == 0 or len(faces) == 0:
        return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name=name)
    
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    mesh_kwargs = {
        'x': x, 'y': y, 'z': z,
        'i': i, 'j': j, 'k': k,
        'name': name,
        'opacity': opacity,
        'hovertemplate': f'<b>{name}</b><br>X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}}<extra></extra>',
        'showscale': face_colors is not None,
    }
    
    if face_colors is not None:
        mesh_kwargs['intensity'] = face_colors
        mesh_kwargs['colorscale'] = 'Viridis'
        if colorbar_title:
            mesh_kwargs['colorbar'] = dict(title=colorbar_title)
    else:
        mesh_kwargs['color'] = color
    
    return go.Mesh3d(**mesh_kwargs)


def visualize_nagata_html(
    filepath: str,
    output_file: Optional[str] = None,
    resolution: int = 10,
    enhance: bool = False,
    scheme: str = 'original',
    color_by_face_id: bool = False,
    show_edges: bool = False,
    opacity_nagata: float = 0.7,
    opacity_enhanced: float = 0.7
):
    """
    生成增强Nagata曲面的交互式HTML可视化
    
    Args:
        filepath: NSM文件路径
        output_file: 输出HTML文件路径，None则自动生成
        resolution: Nagata采样密度
        enhance: 是否启用折纹裂隙修复
        scheme: 法向量计算策略 ('original' 或 'average')
        color_by_face_id: 是否按面片ID着色
        show_edges: 是否显示网格边（Plotly中不支持，仅保留参数）
        opacity_nagata: Nagata曲面透明度
        opacity_enhanced: Enhanced曲面透明度
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: 需要安装plotly库")
        print("  pip install plotly")
        sys.exit(1)
    
    print(f"加载文件: {filepath}")
    mesh_data = load_nsm(filepath)
    
    tri_vertex_normals = mesh_data.tri_vertex_normals
    
    if scheme == 'average':
        tri_vertex_normals = compute_average_normals(
            mesh_data.vertices, mesh_data.triangles, mesh_data.tri_vertex_normals
        )
    
    print(f"计算Nagata曲面 (分辨率={resolution})...")
    
    original_mesh = create_pyvista_mesh(mesh_data)
    orig_verts, orig_faces = polydata_to_arrays(original_mesh)
    
    if enhance:
        print("启用折纹裂隙修复模式...")
        
        eng_path = get_eng_filepath(filepath)
        cached_c_sharps = None
        if has_cached_data(filepath):
            cached_c_sharps = load_enhanced_data(eng_path)
            print(f"使用缓存的增强数据 ({len(cached_c_sharps)} 条裂隙边)")
        
        nagata_mesh = create_nagata_mesh(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution
        )
        
        enhanced_mesh, c_sharps = create_nagata_mesh_enhanced(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution,
            cached_c_sharps=cached_c_sharps
        )
        
        print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
        print(f"Enhanced曲面: {enhanced_mesh.n_points} 个顶点, {enhanced_mesh.n_cells} 个三角形")
        
        nagata_verts, nagata_faces = polydata_to_arrays(nagata_mesh)
        enhanced_verts, enhanced_faces = polydata_to_arrays(enhanced_mesh)
        
        nagata_face_colors = None
        enhanced_face_colors = None
        if color_by_face_id:
            if 'face_id' in nagata_mesh.cell_data:
                nagata_face_colors = nagata_mesh.cell_data['face_id']
            if 'face_id' in enhanced_mesh.cell_data:
                enhanced_face_colors = enhanced_mesh.cell_data['face_id']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mesh', 'Nagata', 'Enhanced'),
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]]
        )
        
        fig.add_trace(
            create_mesh_trace(
                orig_verts, orig_faces,
                name='Mesh', color='lightblue', opacity=1.0,
                face_colors=None
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            create_mesh_trace(
                nagata_verts, nagata_faces,
                name='Nagata', color='lightgreen', opacity=opacity_nagata,
                face_colors=nagata_face_colors,
                colorbar_title='Face ID' if color_by_face_id else ''
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            create_mesh_trace(
                enhanced_verts, enhanced_faces,
                name='Enhanced', color='lightsalmon', opacity=opacity_enhanced,
                face_colors=enhanced_face_colors,
                colorbar_title='Face ID' if color_by_face_id else ''
            ),
            row=1, col=3
        )
        
        all_verts = np.vstack([v for v in [orig_verts, nagata_verts, enhanced_verts] if len(v) > 0])
        if len(all_verts) > 0:
            bbox_min = all_verts.min(axis=0)
            bbox_max = all_verts.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            max_range = np.max(bbox_max - bbox_min) / 2 * 1.1
            
            for col in range(1, 4):
                fig.update_scenes(
                    dict(
                        xaxis=dict(range=[center[0] - max_range, center[0] + max_range], title='X'),
                        yaxis=dict(range=[center[1] - max_range, center[1] + max_range], title='Y'),
                        zaxis=dict(range=[center[2] - max_range, center[2] + max_range], title='Z'),
                        aspectmode='cube',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    row=1, col=col
                )
        
        fig.update_layout(
            title_text=f'Nagata Patch Visualization - {Path(filepath).name}',
            width=1800,
            height=800,
            autosize=True
        )
        
    else:
        nagata_mesh = create_nagata_mesh(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution
        )
        
        print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
        
        nagata_verts, nagata_faces = polydata_to_arrays(nagata_mesh)
        
        nagata_face_colors = None
        if color_by_face_id and 'face_id' in nagata_mesh.cell_data:
            nagata_face_colors = nagata_mesh.cell_data['face_id']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Mesh', 'Nagata'),
            specs=[[{'type': 'scene'}, {'type': 'scene'}]]
        )
        
        fig.add_trace(
            create_mesh_trace(
                orig_verts, orig_faces,
                name='Mesh', color='lightblue', opacity=1.0
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            create_mesh_trace(
                nagata_verts, nagata_faces,
                name='Nagata', color='lightgreen', opacity=opacity_nagata,
                face_colors=nagata_face_colors,
                colorbar_title='Face ID' if color_by_face_id else ''
            ),
            row=1, col=2
        )
        
        all_verts = np.vstack([v for v in [orig_verts, nagata_verts] if len(v) > 0])
        if len(all_verts) > 0:
            bbox_min = all_verts.min(axis=0)
            bbox_max = all_verts.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            max_range = np.max(bbox_max - bbox_min) / 2 * 1.1
            
            for col in range(1, 3):
                fig.update_scenes(
                    dict(
                        xaxis=dict(range=[center[0] - max_range, center[0] + max_range], title='X'),
                        yaxis=dict(range=[center[1] - max_range, center[1] + max_range], title='Y'),
                        zaxis=dict(range=[center[2] - max_range, center[2] + max_range], title='Z'),
                        aspectmode='cube',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    row=1, col=col
                )
        
        fig.update_layout(
            title_text=f'Nagata Patch Visualization - {Path(filepath).name}',
            width=1200,
            height=800,
            autosize=True
        )
    
    if output_file is None:
        base_name = Path(filepath).stem
        mode_str = "enhanced" if enhance else "nagata"
        output_file = f"{base_name}_{mode_str}.html"
    
    fig.write_html(output_file, include_plotlyjs='cdn', full_html=True)
    print(f"\n可视化结果已保存到: {output_file}")
    
    return output_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='生成增强Nagata曲面的交互式HTML可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python visualize_nagata_html.py model.nsm
  
  # 启用增强模式
  python visualize_nagata_html.py model.nsm --enhance
  
  # 调整采样分辨率
  python visualize_nagata_html.py model.nsm --enhance --resolution 20
  
  # 按面片ID着色
  python visualize_nagata_html.py model.nsm --enhance --color-by-face-id
  
  # 指定输出文件
  python visualize_nagata_html.py model.nsm --enhance -o output.html
        """
    )
    
    parser.add_argument('filepath', help='NSM文件路径')
    parser.add_argument('-o', '--output', help='输出HTML文件路径')
    parser.add_argument('-r', '--resolution', type=int, default=10,
                        help='Nagata采样密度 (默认: 10)')
    parser.add_argument('-e', '--enhance', action='store_true',
                        help='启用折纹裂隙修复模式')
    parser.add_argument('-s', '--scheme', choices=['original', 'average'], default='original',
                        help='法向量计算策略 (默认: original)')
    parser.add_argument('--color-by-face-id', action='store_true',
                        help='按面片ID着色')
    parser.add_argument('--opacity-nagata', type=float, default=0.7,
                        help='Nagata曲面透明度 (默认: 0.7)')
    parser.add_argument('--opacity-enhanced', type=float, default=0.7,
                        help='Enhanced曲面透明度 (默认: 0.7)')
    
    args = parser.parse_args()
    
    visualize_nagata_html(
        filepath=args.filepath,
        output_file=args.output,
        resolution=args.resolution,
        enhance=args.enhance,
        scheme=args.scheme,
        color_by_face_id=args.color_by_face_id,
        opacity_nagata=args.opacity_nagata,
        opacity_enhanced=args.opacity_enhanced
    )


if __name__ == "__main__":
    main()
