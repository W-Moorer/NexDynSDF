import numpy as np
import struct
import os

def generate_refined_cone_nsm(radius, height, n_segments, n_height, n_radius, output_path):
    """
    生成高精度离散圆锥模型并保存为NSM文件。
    
    参数:
    - radius: 圆锥底面半径
    - height: 圆锥高度
    - n_segments: 圆周分段数 (环向)
    - n_height: 侧面高度分段数 (纵向)
    - n_radius: 底面半径分段数 (径向)
    """
    
    vertices = []
    triangles = []
    tri_face_ids = []
    
    # --- 1. 生成顶点 ---
    
    # 顶点: (0, 0, height)
    apex_idx = 0
    vertices.append([0.0, 0.0, float(height)])
    
    # 侧面包络面顶点 (不包括顶点)
    # 按从上到下的环排列
    # 每环 n_segments 个点
    side_rings_start = len(vertices)
    for h_idx in range(1, n_height + 1):
        z = height * (1.0 - h_idx / n_height)
        r_curr = radius * (h_idx / n_height)
        for s_idx in range(n_segments):
            theta = 2.0 * np.pi * s_idx / n_segments
            x = r_curr * np.cos(theta)
            y = r_curr * np.sin(theta)
            vertices.append([x, y, z])
            
    # 底面内部顶点 (不归属于侧面，用于底面细分)
    # 中心点: (0, 0, 0)
    base_center_idx = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    
    # 底面中间环 (r从 0 变到 radius，不包括最外环，最外环由侧面底部环提供)
    base_rings_start = len(vertices)
    for r_idx in range(1, n_radius):
        r_curr = radius * (r_idx / n_radius)
        for s_idx in range(n_segments):
            theta = 2.0 * np.pi * s_idx / n_segments
            x = r_curr * np.cos(theta)
            y = r_curr * np.sin(theta)
            vertices.append([x, y, 0.0])
            
    vertices = np.array(vertices, dtype=np.float64)
    
    # --- 2. 生成三角形和 Face ID ---
    
    # A. 侧面三角形 (Face ID: 0)
    # 第一层 (Apex 到底 1 环)
    for s_idx in range(n_segments):
        v1 = side_rings_start + s_idx
        v2 = side_rings_start + (s_idx + 1) % n_segments
        triangles.append([apex_idx, v1, v2])
        tri_face_ids.append(0)
        
    # 其他层 (第 i 环到第 i+1 环)
    for h_idx in range(n_height - 1):
        r1_start = side_rings_start + h_idx * n_segments
        r2_start = side_rings_start + (h_idx + 1) * n_segments
        for s_idx in range(n_segments):
            s_next = (s_idx + 1) % n_segments
            # 两个三角形组成一个四边形
            triangles.append([r1_start + s_idx, r2_start + s_idx, r1_start + s_next])
            tri_face_ids.append(0)
            triangles.append([r1_start + s_next, r2_start + s_idx, r2_start + s_next])
            tri_face_ids.append(0)
            
    # B. 底面三角形 (Face ID: 1)
    # 中心到第 1 环
    for s_idx in range(n_segments):
        v1 = base_rings_start + s_idx
        v2 = base_rings_start + (s_idx + 1) % n_segments
        triangles.append([base_center_idx, v2, v1]) # 顺时针面向下即外法向向下
        tri_face_ids.append(1)
        
    # 中间环到中间环
    for r_idx in range(n_radius - 2):
        r1_start = base_rings_start + r_idx * n_segments
        r2_start = base_rings_start + (r_idx + 1) * n_segments
        for s_idx in range(n_segments):
            s_next = (s_idx + 1) % n_segments
            triangles.append([r1_start + s_idx, r1_start + s_next, r2_start + s_idx])
            tri_face_ids.append(1)
            triangles.append([r1_start + s_next, r2_start + s_next, r2_start + s_idx])
            tri_face_ids.append(1)
            
    # 最后一环到底部边缘环 (由侧面最底部环提供)
    last_base_ring_start = base_rings_start + (n_radius - 2) * n_segments
    bottom_edge_ring_start = side_rings_start + (n_height - 1) * n_segments
    if n_radius > 1:
        for s_idx in range(n_segments):
            s_next = (s_idx + 1) % n_segments
            triangles.append([last_base_ring_start + s_idx, last_base_ring_start + s_next, bottom_edge_ring_start + s_idx])
            tri_face_ids.append(1)
            triangles.append([last_base_ring_start + s_next, bottom_edge_ring_start + s_next, bottom_edge_ring_start + s_idx])
            tri_face_ids.append(1)
    else:
        # 如果 n_radius=1，直接中心连边缘
        for s_idx in range(n_segments):
            v1 = bottom_edge_ring_start + s_idx
            v2 = bottom_edge_ring_start + (s_idx + 1) % n_segments
            triangles.append([base_center_idx, v2, v1])
            tri_face_ids.append(1)

    triangles = np.array(triangles, dtype=np.uint32)
    tri_face_ids = np.array(tri_face_ids, dtype=np.uint32)
    
    # --- 3. 计算解析法向量 ---
    num_triangles = len(triangles)
    tri_vertex_normals = np.zeros((num_triangles, 3, 3), dtype=np.float64)
    
    slope = radius / height
    
    for i in range(num_triangles):
        face_id = tri_face_ids[i]
        tri = triangles[i]
        
        for j in range(3):
            v_idx = tri[j]
            v = vertices[v_idx]
            
            if face_id == 0: # 侧面
                if v_idx == apex_idx:
                    # 顶点处使用面片平均位置的方位角
                    v1 = vertices[tri[1]]
                    v2 = vertices[tri[2]]
                    mid = (v1 + v2) / 2.0
                    rho = np.sqrt(mid[0]**2 + mid[1]**2)
                    if rho < 1e-10: 
                        n = [0, 0, 1] # 退化情况
                    else:
                        n = [mid[0]/rho, mid[1]/rho, slope]
                else:
                    rho = np.sqrt(v[0]**2 + v[1]**2)
                    if rho < 1e-10:
                        n = [0, 0, 1]
                    else:
                        n = [v[0]/rho, v[1]/rho, slope]
                
                n = np.array(n)
                n /= np.linalg.norm(n)
                tri_vertex_normals[i, j] = n
                
            else: # 底面
                tri_vertex_normals[i, j] = [0.0, 0.0, -1.0]
                
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
        
    print(f"导出高精度NSM文件: {output_path}")
    print(f"顶点数: {len(vertices)}, 三角形数: {num_triangles}")

if __name__ == "__main__":
    R = 1.0
    H = 2.0
    N_S = 64  # 圆周分段
    N_H = 10  # 侧面高度分段
    N_R = 10  # 底面径向分段
    output = "e:/workspace/NexDynSDF/output/cone.nsm"
    generate_refined_cone_nsm(R, H, N_S, N_H, N_R, output)
