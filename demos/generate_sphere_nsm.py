import numpy as np
import struct
import os

def generate_sphere_nsm(radius, n_lat, n_lon, output_path):
    """
    生成球体模型并保存为NSM文件。
    
    参数:
    - radius: 球体半径
    - n_lat: 纬度分段数 (从南极到北极)
    - n_lon: 经度分段数 (环绕赤道)
    """
    
    vertices = []
    triangles = []
    tri_face_ids = []
    
    # --- 1. 生成顶点 ---
    
    # 北极点 (0, 0, radius)
    north_pole_idx = 0
    vertices.append([0.0, 0.0, float(radius)])
    
    # 中间纬度环 (不包括南北极)
    # 纬度角 phi 从 pi/n_lat 到 (n_lat-1)*pi/n_lat
    for lat_idx in range(1, n_lat):
        phi = np.pi * lat_idx / n_lat  # 从北极到南极的极角
        z = radius * np.cos(phi)
        r_xy = radius * np.sin(phi)  # 该纬度圈的半径
        
        for lon_idx in range(n_lon):
            theta = 2.0 * np.pi * lon_idx / n_lon  # 经度角
            x = r_xy * np.cos(theta)
            y = r_xy * np.sin(theta)
            vertices.append([x, y, z])
    
    # 南极点 (0, 0, -radius)
    south_pole_idx = len(vertices)
    vertices.append([0.0, 0.0, float(-radius)])
    
    vertices = np.array(vertices, dtype=np.float64)
    
    # --- 2. 生成三角形和 Face ID ---
    
    # 所有三角形都属于同一个面 (Face ID: 0)
    
    # A. 北极附近的三角形 (北极点到第一环)
    first_ring_start = 1
    for lon_idx in range(n_lon):
        v1 = first_ring_start + lon_idx
        v2 = first_ring_start + (lon_idx + 1) % n_lon
        triangles.append([north_pole_idx, v1, v2])
        tri_face_ids.append(0)
    
    # B. 中间环之间的三角形
    for lat_idx in range(n_lat - 2):
        ring1_start = 1 + lat_idx * n_lon
        ring2_start = 1 + (lat_idx + 1) * n_lon
        
        for lon_idx in range(n_lon):
            lon_next = (lon_idx + 1) % n_lon
            
            # 两个三角形组成一个四边形
            # 三角形1: (ring1, lon_idx) -> (ring2, lon_idx) -> (ring1, lon_next)
            triangles.append([ring1_start + lon_idx, ring2_start + lon_idx, ring1_start + lon_next])
            tri_face_ids.append(0)
            
            # 三角形2: (ring1, lon_next) -> (ring2, lon_idx) -> (ring2, lon_next)
            triangles.append([ring1_start + lon_next, ring2_start + lon_idx, ring2_start + lon_next])
            tri_face_ids.append(0)
    
    # C. 南极附近的三角形 (最后一环到南极点)
    last_ring_start = 1 + (n_lat - 2) * n_lon
    for lon_idx in range(n_lon):
        v1 = last_ring_start + lon_idx
        v2 = last_ring_start + (lon_idx + 1) % n_lon
        # 注意顺序，确保法向量朝外
        triangles.append([south_pole_idx, v2, v1])
        tri_face_ids.append(0)
    
    triangles = np.array(triangles, dtype=np.uint32)
    tri_face_ids = np.array(tri_face_ids, dtype=np.uint32)
    
    # --- 3. 计算解析法向量 ---
    # 球体的法向量就是从球心指向顶点位置的单位向量
    num_triangles = len(triangles)
    tri_vertex_normals = np.zeros((num_triangles, 3, 3), dtype=np.float64)
    
    for i in range(num_triangles):
        tri = triangles[i]
        
        for j in range(3):
            v_idx = tri[j]
            v = vertices[v_idx]
            
            # 球体法向量 = 顶点位置 / 半径 (归一化)
            n = np.array(v) / radius
            tri_vertex_normals[i, j] = n
    
    # --- 4. 保存文件 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(b'NSM\x00')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(vertices)))
        f.write(struct.pack('<I', num_triangles))
        f.write(b'\x00' * 48)
        vertices.tofile(f)
        triangles.tofile(f)
        tri_face_ids.tofile(f)
        tri_vertex_normals.tofile(f)
    
    print(f"导出NSM文件: {output_path}")
    print(f"顶点数: {len(vertices)}, 三角形数: {num_triangles}")

if __name__ == "__main__":
    R = 1.0  # 半径
    N_LAT = 16  # 纬度分段数
    N_LON = 16  # 经度分段数
    output = "e:/workspace/NexDynSDF/output/sphere.nsm"
    generate_sphere_nsm(R, N_LAT, N_LON, output)
