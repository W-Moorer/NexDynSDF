#!/usr/bin/env python3
"""
三维SDF零等值面可视化工具 - PyVista版本
支持从SdfLib导出的原始体积数据文件中提取并可视化零等值面

依赖库:
    pip install numpy pyvista

使用方法:
    1. 首先使用SdfSampler工具采样SDF数据:
       SdfSampler.exe gear_test.bin gear_sampled.raw 128

    2. 然后运行此脚本:
       python visualize_sdf_pyvista.py gear_sampled.raw
"""

import struct
import numpy as np
import sys


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
        grid_data = grid_data.reshape((grid_res, grid_res, grid_res))
        
    return {
        'grid': grid_data,
        'resolution': grid_res,
        'bbox_min': np.array(bbox_min),
        'bbox_max': np.array(bbox_max)
    }


def visualize_pyvista(sdf_data_list, titles=None, level=0.0, smooth=True):
    """
    使用PyVista进行3D可视化
    
    参数:
        sdf_data_list: SDF数据列表
        titles: 每个SDF的标题
        level: 等值面值 (默认0表示零等值面)
        smooth: 是否平滑网格
    """
    try:
        import pyvista as pv
    except ImportError:
        print("Error: 需要安装pyvista库")
        print("  pip install pyvista")
        sys.exit(1)
    
    if titles is None:
        titles = [f'SDF {i+1}' for i in range(len(sdf_data_list))]
    
    # 创建绘图窗口
    n_plots = len(sdf_data_list)
    if n_plots == 1:
        plotter = pv.Plotter(window_size=[1200, 900])
        plotters = [plotter]
    else:
        # 多子图布局
        shape = (1, n_plots) if n_plots <= 3 else (2, (n_plots + 1) // 2)
        plotter = pv.Plotter(shape=shape, window_size=[600 * n_plots, 800])
        plotters = []
        for i in range(n_plots):
            if shape[0] == 1:
                plotter.subplot(0, i)
            else:
                row = i // shape[1]
                col = i % shape[1]
                plotter.subplot(row, col)
            plotters.append(plotter)
    
    for idx, (sdf_data, title) in enumerate(zip(sdf_data_list, titles)):
        print(f"\n处理 {title}...")
        print(f"  网格尺寸: {sdf_data['resolution']}x{sdf_data['resolution']}x{sdf_data['resolution']}")
        print(f"  边界框: {sdf_data['bbox_min']} ~ {sdf_data['bbox_max']}")
        
        # 创建UniformGrid
        grid_res = sdf_data['resolution']
        bbox_min = sdf_data['bbox_min']
        bbox_max = sdf_data['bbox_max']
        
        # 计算网格间距
        spacing = (bbox_max - bbox_min) / (grid_res - 1)
        
        # 创建PyVista网格
        grid = pv.ImageData()
        grid.dimensions = np.array([grid_res, grid_res, grid_res])
        grid.spacing = spacing
        grid.origin = bbox_min
        grid.point_data["scalars"] = sdf_data['grid'].flatten(order='F')
        
        # 提取等值面
        print(f"  提取等值面 (level={level})...")
        mesh = grid.contour([level])
        
        if mesh.n_points == 0:
            print(f"  警告: 未找到等值面，尝试调整level值")
            # 尝试自动找到合适的level
            data_min = np.min(sdf_data['grid'])
            data_max = np.max(sdf_data['grid'])
            print(f"  数据范围: [{data_min:.4f}, {data_max:.4f}]")
            continue
        
        print(f"  提取的顶点数: {mesh.n_points}")
        print(f"  提取的面数: {mesh.n_cells}")
        
        # 平滑网格（可选）
        if smooth and mesh.n_points > 0:
            mesh = mesh.smooth(n_iter=100)
        
        # 选择子图
        if n_plots > 1:
            if idx < len(plotters):
                plotter.subplot(0, idx) if n_plots <= 3 else plotter.subplot(idx // 2, idx % 2)
        
        # 添加网格到场景
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        color = colors[idx % len(colors)]
        
        actor = plotter.add_mesh(
            mesh,
            color=color,
            opacity=0.8,
            show_edges=False,
            smooth_shading=True,
            specular=0.5,
            specular_power=20,
        )
        
        # 添加标题
        plotter.add_title(title, font_size=14)
        
        # 添加坐标轴
        plotter.show_axes()
        
        # 设置相机为等比例
        plotter.set_scale(1, 1, 1)
    
    # 添加整体标题
    if n_plots > 1:
        plotter.subplot(0, 0)
    
    # 添加颜色条说明
    plotter.add_text(f"SDF 零等值面 (iso-value = {level})", position='upper_edge', font_size=12)
    
    print("\n启动可视化窗口...")
    print("  操作说明:")
    print("    - 左键拖拽: 旋转")
    print("    - 右键拖拽: 缩放")
    print("    - 中键拖拽: 平移")
    print("    - 滚轮: 缩放")
    print("    - 'q' 或关闭窗口: 退出")
    
    plotter.show()
    
    return plotter


def visualize_volume_slice(sdf_data, title="SDF Volume", axis='z', slice_pos=None):
    """
    使用体数据切片可视化SDF
    
    参数:
        sdf_data: SDF数据
        title: 标题
        axis: 切片轴 ('x', 'y', 'z')
        slice_pos: 切片位置 (None表示中间位置)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("Error: 需要安装pyvista库")
        print("  pip install pyvista")
        sys.exit(1)
    
    print(f"\n处理 {title}...")
    print(f"  网格尺寸: {sdf_data['resolution']}x{sdf_data['resolution']}x{sdf_data['resolution']}")
    
    # 创建UniformGrid
    grid_res = sdf_data['resolution']
    bbox_min = sdf_data['bbox_min']
    bbox_max = sdf_data['bbox_max']
    
    spacing = (bbox_max - bbox_min) / (grid_res - 1)
   # 创建UniformGrid
    grid = pv.ImageData()
    grid.dimensions = np.array([grid_res, grid_res, grid_res])
    grid.spacing = spacing
    grid.origin = bbox_min
    grid.point_data["scalars"] = sdf_data['grid'].flatten(order='F')
    
    # 创建绘图窗口
    plotter = pv.Plotter(window_size=[1200, 900])
    
    # 添加体数据
    plotter.add_volume(
        grid,
        cmap="coolwarm",
        opacity="sigmoid",
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Distance'}
    )
    
    # 添加切片
    if slice_pos is None:
        slice_pos = (bbox_min + bbox_max) / 2
    
    if axis == 'x':
        normal = [1, 0, 0]
        origin = [slice_pos[0], 0, 0]
    elif axis == 'y':
        normal = [0, 1, 0]
        origin = [0, slice_pos[1], 0]
    else:  # 'z'
        normal = [0, 0, 1]
        origin = [0, 0, slice_pos[2]]
    
    slice_mesh = grid.slice(normal=normal, origin=origin)
    plotter.add_mesh(
        slice_mesh,
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={'title': 'Distance'}
    )
    
    # 添加等值面
    try:
        contour = grid.contour([0.0])
        if contour.n_points > 0:
            plotter.add_mesh(
                contour,
                color='white',
                opacity=0.5,
                show_edges=False,
                smooth_shading=True
            )
    except:
        pass
    
    plotter.add_title(title, font_size=14)
    plotter.show_axes()
    plotter.add_text("SDF 体数据可视化", position='upper_edge', font_size=12)
    
    print("\n启动可视化窗口...")
    plotter.show()
    
    return plotter


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用PyVista可视化SDF零等值面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化单个文件
  python visualize_sdf_pyvista.py gear_sampled.raw
  
  # 对比两个文件
  python visualize_sdf_pyvista.py approx_sdf.raw exact_sdf.raw
  
  # 使用不同的等值面值
  python visualize_sdf_pyvista.py gear_sampled.raw -l 0.1
  
  # 体数据切片可视化
  python visualize_sdf_pyvista.py gear_sampled.raw --volume
        """
    )
    
    parser.add_argument('files', nargs='+', help='SDF原始数据文件路径')
    parser.add_argument('-l', '--level', type=float, default=0.0, 
                        help='等值面值 (默认: 0.0)')
    parser.add_argument('-t', '--titles', nargs='+', 
                        help='每个SDF的标题')
    parser.add_argument('--no-smooth', action='store_true',
                        help='禁用网格平滑')
    parser.add_argument('--volume', action='store_true',
                        help='使用体数据切片模式可视化')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='z',
                        help='切片轴 (默认: z)')
    
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
        titles = [f'SDF {i+1}' for i in range(len(sdf_data_list))]
    
    # 可视化
    if args.volume:
        # 体数据切片模式
        for idx, (sdf_data, title) in enumerate(zip(sdf_data_list, titles)):
            visualize_volume_slice(sdf_data, title, args.axis)
    else:
        # 等值面模式
        visualize_pyvista(
            sdf_data_list, 
            titles, 
            args.level, 
            smooth=not args.no_smooth
        )


if __name__ == '__main__':
    main()
