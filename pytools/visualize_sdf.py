#!/usr/bin/env python3
"""
三维SDF零等值面可视化工具
支持从SdfLib导出的原始体积数据文件中提取并可视化零等值面

依赖库:
    pip install numpy scipy matplotlib plotly

使用方法:
    1. 首先使用SdfSampler工具采样SDF数据:
       SdfSampler.exe Gear_ApproxSDF_High.bin approx_sdf.raw 128
       SdfSampler.exe Gear_ExactSDF_High.bin exact_sdf.raw 128

    2. 然后运行此脚本:
       python visualize_sdf.py approx_sdf.raw exact_sdf.raw
"""

import struct
import numpy as np
from scipy.ndimage import zoom
import sys
import os

def _compute_component_labels(vertices_world, faces):
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        print("Error: 需要安装scipy库以进行连通域分析")
        sys.exit(1)

    if vertices_world.size == 0 or faces.size == 0:
        return None, None, None, None, None

    n_vertices = int(vertices_world.shape[0])
    edges01 = faces[:, [0, 1]]
    edges12 = faces[:, [1, 2]]
    edges20 = faces[:, [2, 0]]
    edges = np.concatenate([edges01, edges12, edges20], axis=0).astype(np.int64, copy=False)
    rows = np.concatenate([edges[:, 0], edges[:, 1]], axis=0)
    cols = np.concatenate([edges[:, 1], edges[:, 0]], axis=0)
    data = np.ones(rows.shape[0], dtype=np.uint8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    if n_components <= 1:
        return n_components, labels, None, None, None

    vertex_counts = np.bincount(labels, minlength=n_components)
    face_labels = labels[faces[:, 0]]
    face_counts = np.bincount(face_labels, minlength=n_components)

    main_component = int(np.argmax(vertex_counts))
    return n_components, labels, vertex_counts, face_counts, main_component

def _component_info(vertices_world, idxs, vertex_count, face_count, rng):
    verts = vertices_world[idxs]
    centroid = verts.mean(axis=0)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)

    if idxs.size > 200000:
        sample = rng.choice(idxs, size=200000, replace=False)
        diffs = vertices_world[sample] - centroid
        nearest = int(sample[np.argmin(np.einsum('ij,ij->i', diffs, diffs))])
    else:
        diffs = verts - centroid
        nearest = int(idxs[np.argmin(np.einsum('ij,ij->i', diffs, diffs))])

    rep = vertices_world[nearest]
    return {
        'vertex_count': int(vertex_count),
        'face_count': int(face_count),
        'centroid': centroid.astype(np.float64, copy=False),
        'bbox_min': bbox_min.astype(np.float64, copy=False),
        'bbox_max': bbox_max.astype(np.float64, copy=False),
        'point': rep.astype(np.float64, copy=False)
    }

def _extract_splash_components(vertices_world, faces, min_component_vertices=50, min_component_faces=50):
    n_components, labels, vertex_counts, face_counts, main_component = _compute_component_labels(
        vertices_world, faces
    )
    if n_components is None or n_components <= 0:
        return None, []

    rng = np.random.default_rng(0)
    components = []
    for cid in range(n_components):
        if cid == main_component:
            continue
        if vertex_counts is None or face_counts is None:
            continue
        if vertex_counts[cid] < int(min_component_vertices):
            continue
        if face_counts[cid] < int(min_component_faces):
            continue
        idxs = np.nonzero(labels == cid)[0]
        if idxs.size == 0:
            continue
        info = _component_info(
            vertices_world=vertices_world,
            idxs=idxs,
            vertex_count=vertex_counts[cid],
            face_count=face_counts[cid],
            rng=rng
        )
        info['component_id'] = int(cid)
        components.append(info)

    if vertex_counts is None or face_counts is None:
        main_info = None
    else:
        main_idxs = np.nonzero(labels == main_component)[0]
        main_info = _component_info(
            vertices_world=vertices_world,
            idxs=main_idxs,
            vertex_count=vertex_counts[main_component],
            face_count=face_counts[main_component],
            rng=rng
        )
        main_info['component_id'] = int(main_component)

    components.sort(key=lambda c: c['vertex_count'], reverse=True)
    return main_info, components

def _extract_splash_points(vertices_world, faces, max_points=10, min_component_vertices=50, min_component_faces=50):
    main_info, components = _extract_splash_components(
        vertices_world=vertices_world,
        faces=faces,
        min_component_vertices=min_component_vertices,
        min_component_faces=min_component_faces
    )
    if not components:
        return []
    components = components[: int(max_points)]
    points = []
    for c in components:
        points.append({
            'component_id': int(c['component_id']),
            'vertex_count': int(c['vertex_count']),
            'face_count': int(c['face_count']),
            'point': c['point']
        })
    return points

def load_raw_sdf(filepath):
    """
    加载SdfSampler生成的原始SDF数据文件
    
    文件格式:
        - 4字节: grid_resolution (int)
        - 4字节: bbox_min_x (float)
        - 4字节: bbox_min_y (float)
        - 4字节: bbox_min_z (float)
        - 4字节: bbox_max_x (float)
        - 4字节: bbox_max_y (float)
        - 4字节: bbox_max_z (float)
        - 剩余: grid_data (float array)
    """
    with open(filepath, 'rb') as f:
        # 读取头部信息
        grid_res = struct.unpack('i', f.read(4))[0]
        bbox_min = struct.unpack('fff', f.read(12))
        bbox_max = struct.unpack('fff', f.read(12))
        
        # 读取网格数据
        num_voxels = grid_res * grid_res * grid_res
        grid_data = np.frombuffer(f.read(num_voxels * 4), dtype=np.float32)
        grid_data = grid_data.reshape((grid_res, grid_res, grid_res), order='F')
        
    return {
        'grid': grid_data,
        'resolution': grid_res,
        'bbox_min': np.array(bbox_min),
        'bbox_max': np.array(bbox_max)
    }


def marching_cubes_extract(grid_data, level=0.0):
    """
    使用Marching Cubes算法从体数据中提取等值面
    
    参数:
        grid_data: 3D numpy数组
        level: 等值面值 (默认0表示零等值面)
    
    返回:
        vertices: 顶点数组 (N, 3)
        faces: 面索引数组 (M, 3)
    """
    try:
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        
        # 可选：对数据进行轻微平滑以减少噪声
        # smoothed = gaussian_filter(grid_data, sigma=0.5)
        
        # 提取等值面（确保数组可写）
        grid_copy = np.array(grid_data, copy=True)
        vertices, faces, normals, values = marching_cubes(grid_copy, level=level)
        
        return vertices, faces, normals
        
    except ImportError:
        print("Error: 需要安装scikit-image库")
        print("  pip install scikit-image")
        sys.exit(1)


def visualize_matplotlib(sdf_data_list, titles=None, level=0.0):
    """
    使用Matplotlib进行3D可视化（静态）
    
    参数:
        sdf_data_list: SDF数据列表
        titles: 每个SDF的标题
        level: 等值面值
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    if titles is None:
        titles = [f'SDF {i+1}' for i in range(len(sdf_data_list))]
    
    fig = plt.figure(figsize=(12, 6 * len(sdf_data_list)))
    
    for idx, (sdf_data, title) in enumerate(zip(sdf_data_list, titles)):
        print(f"\n处理 {title}...")
        print(f"  网格尺寸: {sdf_data['resolution']}x{sdf_data['resolution']}x{sdf_data['resolution']}")
        print(f"  边界框: {sdf_data['bbox_min']} ~ {sdf_data['bbox_max']}")
        
        # 提取等值面
        vertices, faces, normals = marching_cubes_extract(sdf_data['grid'], level)
        
        print(f"  提取的顶点数: {len(vertices)}")
        print(f"  提取的面数: {len(faces)}")
        
        # 将顶点从网格坐标转换到世界坐标
        bbox_min = sdf_data['bbox_min']
        bbox_max = sdf_data['bbox_max']
        grid_res = sdf_data['resolution']
        
        spacing = (bbox_max - bbox_min) / (grid_res - 1)
        vertices_world = bbox_min + vertices * spacing
        
        # 创建3D图
        ax = fig.add_subplot(len(sdf_data_list), 1, idx + 1, projection='3d')
        
        # 创建面片集合
        mesh = Poly3DCollection(vertices_world[faces], alpha=0.7, edgecolor='none')
        mesh.set_facecolor('lightblue')
        mesh.set_edgecolor('darkblue')
        ax.add_collection3d(mesh)
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title} - 零等值面 (vertices: {len(vertices)})')
        
        # 设置显示范围
        ax.set_xlim(bbox_min[0], bbox_max[0])
        ax.set_ylim(bbox_min[1], bbox_max[1])
        ax.set_zlim(bbox_min[2], bbox_max[2])
        
        # 设置等比例
        max_range = np.max(bbox_max - bbox_min)
        mid_x = (bbox_max[0] + bbox_min[0]) * 0.5
        mid_y = (bbox_max[1] + bbox_min[1]) * 0.5
        mid_z = (bbox_max[2] + bbox_min[2]) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    plt.tight_layout()
    plt.savefig('sdf_isosurface.png', dpi=150, bbox_inches='tight')
    print(f"\n图像已保存到: sdf_isosurface.png")
    plt.show()


def visualize_plotly(sdf_data_list, titles=None, level=0.0):
    """
    使用Plotly进行交互式3D可视化
    
    参数:
        sdf_data_list: SDF数据列表
        titles: 每个SDF的标题
        level: 等值面值
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: 需要安装plotly库")
        print("  pip install plotly")
        sys.exit(1)
    
    if titles is None:
        titles = [f'SDF {i+1}' for i in range(len(sdf_data_list))]
    
    if len(sdf_data_list) == 1:
        fig = go.Figure()
    else:
        fig = make_subplots(
            rows=1, cols=len(sdf_data_list),
            subplot_titles=titles,
            specs=[[{'type': 'surface'} for _ in range(len(sdf_data_list))]]
        )
    
    for idx, (sdf_data, title) in enumerate(zip(sdf_data_list, titles)):
        print(f"\n处理 {title}...")
        print(f"  网格尺寸: {sdf_data['resolution']}x{sdf_data['resolution']}x{sdf_data['resolution']}")
        print(f"  边界框: {sdf_data['bbox_min']} ~ {sdf_data['bbox_max']}")
        
        # 提取等值面
        vertices, faces, normals = marching_cubes_extract(sdf_data['grid'], level)
        
        print(f"  提取的顶点数: {len(vertices)}")
        print(f"  提取的面数: {len(faces)}")
        
        # 将顶点从网格坐标转换到世界坐标
        bbox_min = sdf_data['bbox_min']
        bbox_max = sdf_data['bbox_max']
        grid_res = sdf_data['resolution']
        
        spacing = (bbox_max - bbox_min) / (grid_res - 1)
        vertices_world = bbox_min + vertices * spacing
        
        # 创建三角网格
        x, y, z = vertices_world[:, 0], vertices_world[:, 1], vertices_world[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        # 计算每个面的颜色（基于法线）
        # 处理可能的NaN/Inf
        valid_normals = np.nan_to_num(normals, nan=0.0, posinf=1.0, neginf=-1.0)
        colors = np.abs(valid_normals[faces[:, 0]])
        face_colors = ['rgb({},{},{})'.format(
            int(255 * (0.5 + 0.5 * c[0])),
            int(255 * (0.5 + 0.5 * c[1])),
            int(255 * (0.5 + 0.5 * c[2]))
        ) for c in colors]
        
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.9,
            color='lightblue',
            colorscale='Viridis',
            showscale=False,
            name=title,
            hovertemplate=f'<b>{title}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
        )
        
        if len(sdf_data_list) == 1:
            fig.add_trace(mesh)
        else:
            fig.add_trace(mesh, row=1, col=idx + 1)
    
    # 更新布局
    fig.update_layout(
        title_text=f'SDF 零等值面可视化 (iso-value = {level})',
        width=800 * len(sdf_data_list),
        height=800,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        )
    )
    
    if len(sdf_data_list) > 1:
        for i in range(len(sdf_data_list)):
            fig.update_scenes(
                aspectmode='cube',
                row=1, col=i+1
            )
    
    # 保存为HTML文件
    html_file = 'sdf_isosurface_interactive.html'
    fig.write_html(html_file)
    print(f"\n交互式HTML已保存到: {html_file}")
    
    # 显示图形
    fig.show()


def visualize_comparison(sdf_data_1, sdf_data_2, title1="SDF 1", title2="SDF 2", level=0.0):
    """
    并排比较两个SDF的零等值面
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: 需要安装plotly库")
        print("  pip install plotly")
        sys.exit(1)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title1, title2),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    colors = ['lightblue', 'lightgreen']
    
    for idx, (sdf_data, title, color) in enumerate(zip([sdf_data_1, sdf_data_2], [title1, title2], colors)):
        print(f"\n处理 {title}...")
        print(f"  网格尺寸: {sdf_data['resolution']}x{sdf_data['resolution']}x{sdf_data['resolution']}")
        
        vertices, faces, normals = marching_cubes_extract(sdf_data['grid'], level)
        print(f"  提取的顶点数: {len(vertices)}, 面数: {len(faces)}")
        
        bbox_min = sdf_data['bbox_min']
        bbox_max = sdf_data['bbox_max']
        grid_res = sdf_data['resolution']
        spacing = (bbox_max - bbox_min) / (grid_res - 1)
        vertices_world = bbox_min + vertices * spacing
        
        x, y, z = vertices_world[:, 0], vertices_world[:, 1], vertices_world[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.8,
            color=color,
            name=title,
            hovertemplate=f'<b>{title}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
        )
        
        fig.add_trace(mesh, row=1, col=idx + 1)
    
    fig.update_layout(
        title_text=f'SDF 零等值面对比 (iso-value = {level})',
        width=1600,
        height=800
    )
    
    for i in range(2):
        fig.update_scenes(
            aspectmode='cube',
            row=1, col=i+1
        )
    
    html_file = 'sdf_comparison.html'
    fig.write_html(html_file)
    print(f"\n对比HTML已保存到: {html_file}")
    fig.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='可视化SDF零等值面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化单个文件
  python visualize_sdf.py approx_sdf.raw
  
  # 对比两个文件
  python visualize_sdf.py approx_sdf.raw exact_sdf.raw
  
  # 使用不同的等值面值
  python visualize_sdf.py approx_sdf.raw -l 0.1
        """
    )
    
    parser.add_argument('files', nargs='+', help='SDF原始数据文件路径')
    parser.add_argument('-l', '--level', type=float, default=0.0, 
                        help='等值面值 (默认: 0.0)')
    parser.add_argument('-t', '--titles', nargs='+', 
                        help='每个SDF的标题')
    parser.add_argument('-s', '--static', action='store_true',
                        help='使用静态Matplotlib可视化而非交互式Plotly')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='对比模式：并排显示两个SDF')
    parser.add_argument('--splash-points', type=int, default=0,
                        help='输出飞溅碎片代表点数量（0表示不输出）')
    parser.add_argument('--min-splash-verts', type=int, default=50,
                        help='飞溅连通域最少顶点数')
    parser.add_argument('--min-splash-faces', type=int, default=50,
                        help='飞溅连通域最少面数')
    parser.add_argument('--splash-out', type=str, default='',
                        help='将飞溅代表点写入文本文件')
    parser.add_argument('--splash-report', type=str, default='',
                        help='将飞溅连通域详情写入文本文件')
    parser.add_argument('--no-visualize', action='store_true',
                        help='仅输出飞溅代表点，不进行可视化')
    
    args = parser.parse_args()
    
    # 加载所有SDF文件
    sdf_data_list = []
    for filepath in args.files:
        print(f"\n加载文件: {filepath}")
        try:
            sdf_data = load_raw_sdf(filepath)
            sdf_data_list.append(sdf_data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            sys.exit(1)
    
    # 设置标题
    titles = args.titles
    if titles is None:
        if args.compare and len(sdf_data_list) == 2:
            titles = ["Approximate SDF", "Exact SDF"]
        else:
            titles = [f'SDF {i+1}' for i in range(len(sdf_data_list))]
    
    if args.splash_points > 0:
        out_lines = ["component_id vertex_count face_count x y z"]
        for sdf_data, title in zip(sdf_data_list, titles):
            vertices, faces, _normals = marching_cubes_extract(sdf_data['grid'], args.level)
            bbox_min = sdf_data['bbox_min']
            bbox_max = sdf_data['bbox_max']
            grid_res = sdf_data['resolution']
            spacing = (bbox_max - bbox_min) / (grid_res - 1)
            vertices_world = bbox_min + vertices * spacing

            points = _extract_splash_points(
                vertices_world=vertices_world,
                faces=faces,
                max_points=args.splash_points,
                min_component_vertices=args.min_splash_verts,
                min_component_faces=args.min_splash_faces
            )

            print(f"\n{title} 飞溅连通域代表点（排除最大连通域）:")
            if not points:
                print("  (无)")
                continue
            for p in points:
                x, y, z = p['point'].tolist()
                line = f"{p['component_id']} {p['vertex_count']} {p['face_count']} {x:.6f} {y:.6f} {z:.6f}"
                print("  " + line)
                out_lines.append(line)

        if args.splash_out:
            with open(args.splash_out, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            print(f"\n飞溅代表点已写入: {args.splash_out}")

    if args.splash_report:
        report_lines = [
            "file title component_id is_main vertex_count face_count "
            "bbox_min_x bbox_min_y bbox_min_z bbox_max_x bbox_max_y bbox_max_z "
            "centroid_x centroid_y centroid_z rep_x rep_y rep_z"
        ]
        for sdf_data, title, filepath in zip(sdf_data_list, titles, args.files):
            vertices, faces, _normals = marching_cubes_extract(sdf_data['grid'], args.level)
            bbox_min = sdf_data['bbox_min']
            bbox_max = sdf_data['bbox_max']
            grid_res = sdf_data['resolution']
            spacing = (bbox_max - bbox_min) / (grid_res - 1)
            vertices_world = bbox_min + vertices * spacing

            main_info, components = _extract_splash_components(
                vertices_world=vertices_world,
                faces=faces,
                min_component_vertices=args.min_splash_verts,
                min_component_faces=args.min_splash_faces
            )

            basename = os.path.basename(filepath)
            if main_info is not None:
                line = (
                    f"{basename} {title} {main_info['component_id']} 1 "
                    f"{main_info['vertex_count']} {main_info['face_count']} "
                    f"{main_info['bbox_min'][0]:.6f} {main_info['bbox_min'][1]:.6f} {main_info['bbox_min'][2]:.6f} "
                    f"{main_info['bbox_max'][0]:.6f} {main_info['bbox_max'][1]:.6f} {main_info['bbox_max'][2]:.6f} "
                    f"{main_info['centroid'][0]:.6f} {main_info['centroid'][1]:.6f} {main_info['centroid'][2]:.6f} "
                    f"{main_info['point'][0]:.6f} {main_info['point'][1]:.6f} {main_info['point'][2]:.6f}"
                )
                report_lines.append(line)

            for c in components:
                line = (
                    f"{basename} {title} {c['component_id']} 0 "
                    f"{c['vertex_count']} {c['face_count']} "
                    f"{c['bbox_min'][0]:.6f} {c['bbox_min'][1]:.6f} {c['bbox_min'][2]:.6f} "
                    f"{c['bbox_max'][0]:.6f} {c['bbox_max'][1]:.6f} {c['bbox_max'][2]:.6f} "
                    f"{c['centroid'][0]:.6f} {c['centroid'][1]:.6f} {c['centroid'][2]:.6f} "
                    f"{c['point'][0]:.6f} {c['point'][1]:.6f} {c['point'][2]:.6f}"
                )
                report_lines.append(line)

        with open(args.splash_report, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\n飞溅连通域详情已写入: {args.splash_report}")

    if args.no_visualize:
        return

    if args.compare and len(sdf_data_list) == 2:
        visualize_comparison(sdf_data_list[0], sdf_data_list[1],
                            titles[0], titles[1], args.level)
    elif args.static:
        visualize_matplotlib(sdf_data_list, titles, args.level)
    else:
        visualize_plotly(sdf_data_list, titles, args.level)


if __name__ == '__main__':
    main()
