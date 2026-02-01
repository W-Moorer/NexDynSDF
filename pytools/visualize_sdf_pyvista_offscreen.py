#!/usr/bin/env python3
"""
三维SDF零等值面可视化工具 - PyVista版本 (离屏渲染)
用于生成静态图片而不打开交互窗口

依赖库:
    pip install numpy pyvista

使用方法:
    python visualize_sdf_pyvista_offscreen.py gear_sampled.raw -o gear_vis.png
"""

import struct
import numpy as np
import sys
import os

# 设置离屏渲染
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'false'

import pyvista as pv
pv.OFF_SCREEN = True


def load_raw_sdf(filepath):
    """加载SdfSampler生成的原始SDF数据文件"""
    with open(filepath, 'rb') as f:
        grid_res = struct.unpack('i', f.read(4))[0]
        bbox_min = struct.unpack('fff', f.read(12))
        bbox_max = struct.unpack('fff', f.read(12))
        num_voxels = grid_res * grid_res * grid_res
        grid_data = np.frombuffer(f.read(num_voxels * 4), dtype=np.float32)
        grid_data = grid_data.reshape((grid_res, grid_res, grid_res))
        
    return {
        'grid': grid_data,
        'resolution': grid_res,
        'bbox_min': np.array(bbox_min),
        'bbox_max': np.array(bbox_max)
    }


def visualize_and_save(sdf_data, output_path, title="SDF", level=0.0):
    """
    使用PyVista进行3D可视化并保存为图片
    """
    print(f"\n处理 {title}...")
    print(f"  网格尺寸: {sdf_data['resolution']}x{sdf_data['resolution']}x{sdf_data['resolution']}")
    print(f"  边界框: {sdf_data['bbox_min']} ~ {sdf_data['bbox_max']}")
    
    # 创建UniformGrid
    grid_res = sdf_data['resolution']
    bbox_min = sdf_data['bbox_min']
    bbox_max = sdf_data['bbox_max']
    
    spacing = (bbox_max - bbox_min) / (grid_res - 1)
    
    grid = pv.ImageData()
    grid.dimensions = np.array([grid_res, grid_res, grid_res])
    grid.spacing = spacing
    grid.origin = bbox_min
    grid.point_data["scalars"] = sdf_data['grid'].flatten(order='F')
    
    # 提取等值面
    print(f"  提取等值面 (level={level})...")
    mesh = grid.contour([level])
    
    if mesh.n_points == 0:
        print(f"  警告: 未找到等值面")
        data_min = np.min(sdf_data['grid'])
        data_max = np.max(sdf_data['grid'])
        print(f"  数据范围: [{data_min:.4f}, {data_max:.4f}]")
        return False
    
    print(f"  提取的顶点数: {mesh.n_points}")
    print(f"  提取的面数: {mesh.n_cells}")
    
    # 平滑网格
    mesh = mesh.smooth(n_iter=50)
    
    # 创建绘图窗口
    plotter = pv.Plotter(window_size=[1200, 900], off_screen=True)
    
    # 添加网格
    plotter.add_mesh(
        mesh,
        color='lightblue',
        opacity=0.9,
        show_edges=False,
        smooth_shading=True,
        specular=0.5,
        specular_power=20,
    )
    
    # 添加标题和坐标轴
    plotter.add_title(f"{title} - 零等值面", font_size=16)
    plotter.show_axes()
    
    # 设置相机视角
    plotter.camera_position = 'iso'
    plotter.set_scale(1, 1, 1)
    
    # 保存图片
    plotter.screenshot(output_path, transparent_background=False)
    print(f"  图片已保存: {output_path}")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用PyVista可视化SDF零等值面（离屏渲染）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_sdf_pyvista_offscreen.py gear_sampled.raw -o gear.png
  python visualize_sdf_pyvista_offscreen.py sphere_sampled.raw -o sphere.png -l 0.0
        """
    )
    
    parser.add_argument('file', help='SDF原始数据文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出图片路径')
    parser.add_argument('-l', '--level', type=float, default=0.0, 
                        help='等值面值 (默认: 0.0)')
    parser.add_argument('-t', '--title', default='SDF', help='标题')
    
    args = parser.parse_args()
    
    # 加载SDF文件
    print(f"加载文件: {args.file}")
    try:
        sdf_data = load_raw_sdf(args.file)
    except Exception as e:
        print(f"Error loading {args.file}: {e}")
        sys.exit(1)
    
    # 可视化并保存
    success = visualize_and_save(sdf_data, args.output, args.title, args.level)
    
    if success:
        print("\n可视化完成!")
        sys.exit(0)
    else:
        print("\n可视化失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()
