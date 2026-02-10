import numpy as np
import struct
import os

def generate_box_nsm(length_x, length_y, length_z, n_segments_xy, n_segments_z, output_path):
    """
    生成长方体模型并保存为NSM文件。
    
    参数:
    - length_x: X方向长度
    - length_y: Y方向长度
    - length_z: Z方向长度
    - n_segments_xy: XY平面的分段数 (用于上下大面)
    - n_segments_z: Z方向的分段数 (用于侧面)
    """
    
    vertices = []
    triangles = []
    tri_face_ids = []
    
    half_x = length_x / 2.0
    half_y = length_y / 2.0
    half_z = length_z / 2.0
    
    # --- 1. 生成每个面的顶点和三角形 ---
    
    # 长方体6个面的定义 (法向量朝外)
    # 每个面: (中心位置, 法向量, 切向1, 切向2, 分段数1, 分段数2)
    faces = [
        # 右面 (+X)
        {
            'id': 0,
            'center': [half_x, 0, 0],
            'normal': [1, 0, 0],
            'tangent1': [0, 1, 0],  # Y方向
            'tangent2': [0, 0, 1],  # Z方向
            'n1': n_segments_xy,    # Y方向分段
            'n2': n_segments_z,     # Z方向分段
            'len1': length_y,
            'len2': length_z,
        },
        # 左面 (-X)
        {
            'id': 1,
            'center': [-half_x, 0, 0],
            'normal': [-1, 0, 0],
            'tangent1': [0, 0, 1],  # Z方向
            'tangent2': [0, 1, 0],  # Y方向
            'n1': n_segments_z,
            'n2': n_segments_xy,
            'len1': length_z,
            'len2': length_y,
        },
        # 上面 (+Y)
        {
            'id': 2,
            'center': [0, half_y, 0],
            'normal': [0, 1, 0],
            'tangent1': [0, 0, 1],  # Z方向
            'tangent2': [1, 0, 0],  # X方向
            'n1': n_segments_z,
            'n2': n_segments_xy,
            'len1': length_z,
            'len2': length_x,
        },
        # 下面 (-Y)
        {
            'id': 3,
            'center': [0, -half_y, 0],
            'normal': [0, -1, 0],
            'tangent1': [1, 0, 0],  # X方向
            'tangent2': [0, 0, 1],  # Z方向
            'n1': n_segments_xy,
            'n2': n_segments_z,
            'len1': length_x,
            'len2': length_z,
        },
        # 前面 (+Z) - 大面
        {
            'id': 4,
            'center': [0, 0, half_z],
            'normal': [0, 0, 1],
            'tangent1': [1, 0, 0],  # X方向
            'tangent2': [0, 1, 0],  # Y方向
            'n1': n_segments_xy,
            'n2': n_segments_xy,
            'len1': length_x,
            'len2': length_y,
        },
        # 后面 (-Z) - 大面
        {
            'id': 5,
            'center': [0, 0, -half_z],
            'normal': [0, 0, -1],
            'tangent1': [0, 1, 0],  # Y方向
            'tangent2': [1, 0, 0],  # X方向
            'n1': n_segments_xy,
            'n2': n_segments_xy,
            'len1': length_y,
            'len2': length_x,
        },
    ]
    
    # 为每个面生成网格
    for face in faces:
        face_id = face['id']
        center = np.array(face['center'])
        normal = np.array(face['normal'])
        t1 = np.array(face['tangent1'])
        t2 = np.array(face['tangent2'])
        n1 = face['n1']
        n2 = face['n2']
        len1 = face['len1']
        len2 = face['len2']
        
        # 记录该面的起始顶点索引
        face_vertex_start = len(vertices)
        
        # 生成该面的顶点 (n1+1) x (n2+1) 的网格
        for i in range(n1 + 1):
            for j in range(n2 + 1):
                # 参数坐标 [-len1/2, len1/2] 和 [-len2/2, len2/2]
                u = -len1 / 2 + len1 * i / n1
                v = -len2 / 2 + len2 * j / n2
                
                # 顶点位置 = 中心 + u * t1 + v * t2
                pos = center + u * t1 + v * t2
                vertices.append(pos.tolist())
        
        # 生成该面的三角形
        for i in range(n1):
            for j in range(n2):
                # 当前四边形的四个顶点索引
                v00 = face_vertex_start + i * (n2 + 1) + j
                v01 = face_vertex_start + i * (n2 + 1) + (j + 1)
                v10 = face_vertex_start + (i + 1) * (n2 + 1) + j
                v11 = face_vertex_start + (i + 1) * (n2 + 1) + (j + 1)
                
                # 两个三角形组成一个四边形
                # 三角形1: v00 -> v10 -> v01
                triangles.append([v00, v10, v01])
                tri_face_ids.append(face_id)
                
                # 三角形2: v01 -> v10 -> v11
                triangles.append([v01, v10, v11])
                tri_face_ids.append(face_id)
    
    vertices = np.array(vertices, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.uint32)
    tri_face_ids = np.array(tri_face_ids, dtype=np.uint32)
    
    # --- 2. 计算法向量 ---
    # 长方体每个面的法向量是固定的
    face_normals = np.array([
        [1, 0, 0],   # 右面 (+X)
        [-1, 0, 0],  # 左面 (-X)
        [0, 1, 0],   # 上面 (+Y)
        [0, -1, 0],  # 下面 (-Y)
        [0, 0, 1],   # 前面 (+Z)
        [0, 0, -1],  # 后面 (-Z)
    ], dtype=np.float64)
    
    num_triangles = len(triangles)
    tri_vertex_normals = np.zeros((num_triangles, 3, 3), dtype=np.float64)
    
    for i in range(num_triangles):
        face_id = tri_face_ids[i]
        normal = face_normals[face_id]
        
        # 该三角形的三个顶点使用相同的法向量
        for j in range(3):
            tri_vertex_normals[i, j] = normal
    
    # --- 3. 保存文件 ---
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
    L_X = 5.0   # X方向长度
    L_Y = 5.0   # Y方向长度
    L_Z = 0.5   # Z方向高度 (薄板)
    N_XY = 4    # XY平面分段数 (大面)
    N_Z = 2     # Z方向分段数 (侧面)
    output = "e:/workspace/NexDynSDF/output/box.nsm"
    generate_box_nsm(L_X, L_Y, L_Z, N_XY, N_Z, output)
