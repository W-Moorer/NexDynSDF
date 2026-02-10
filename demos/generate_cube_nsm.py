import numpy as np
import struct
import os

def generate_cube_nsm(half_size, n_segments, output_path):
    """
    生成正方体模型并保存为NSM文件。
    
    参数:
    - half_size: 半边长 (质心在原点，所以顶点坐标范围是 [-half_size, half_size])
    - n_segments: 每个面的分段数 (每个面会被分成 n_segments x n_segments 个小正方形)
    """
    
    vertices = []
    triangles = []
    tri_face_ids = []
    
    # --- 1. 生成每个面的顶点和三角形 ---
    
    # 正方体6个面的定义 (法向量朝外)
    # 每个面: (中心位置, 法向量, 切向1, 切向2)
    faces = [
        # 右面 (+X)
        {
            'id': 0,
            'center': [half_size, 0, 0],
            'normal': [1, 0, 0],
            'tangent1': [0, 1, 0],  # Y方向
            'tangent2': [0, 0, 1],  # Z方向
        },
        # 左面 (-X)
        {
            'id': 1,
            'center': [-half_size, 0, 0],
            'normal': [-1, 0, 0],
            'tangent1': [0, 0, 1],  # Z方向
            'tangent2': [0, 1, 0],  # Y方向
        },
        # 上面 (+Y)
        {
            'id': 2,
            'center': [0, half_size, 0],
            'normal': [0, 1, 0],
            'tangent1': [0, 0, 1],  # Z方向
            'tangent2': [1, 0, 0],  # X方向
        },
        # 下面 (-Y)
        {
            'id': 3,
            'center': [0, -half_size, 0],
            'normal': [0, -1, 0],
            'tangent1': [1, 0, 0],  # X方向
            'tangent2': [0, 0, 1],  # Z方向
        },
        # 前面 (+Z)
        {
            'id': 4,
            'center': [0, 0, half_size],
            'normal': [0, 0, 1],
            'tangent1': [1, 0, 0],  # X方向
            'tangent2': [0, 1, 0],  # Y方向
        },
        # 后面 (-Z)
        {
            'id': 5,
            'center': [0, 0, -half_size],
            'normal': [0, 0, -1],
            'tangent1': [0, 1, 0],  # Y方向
            'tangent2': [1, 0, 0],  # X方向
        },
    ]
    
    # 为每个面生成网格
    for face in faces:
        face_id = face['id']
        center = np.array(face['center'])
        normal = np.array(face['normal'])
        t1 = np.array(face['tangent1'])
        t2 = np.array(face['tangent2'])
        
        # 记录该面的起始顶点索引
        face_vertex_start = len(vertices)
        
        # 生成该面的顶点 (n_segments+1) x (n_segments+1) 的网格
        for i in range(n_segments + 1):
            for j in range(n_segments + 1):
                # 参数坐标 [-half_size, half_size]
                u = -half_size + 2 * half_size * i / n_segments
                v = -half_size + 2 * half_size * j / n_segments
                
                # 顶点位置 = 中心 + u * t1 + v * t2
                pos = center + u * t1 + v * t2
                vertices.append(pos.tolist())
        
        # 生成该面的三角形
        for i in range(n_segments):
            for j in range(n_segments):
                # 当前四边形的四个顶点索引
                v00 = face_vertex_start + i * (n_segments + 1) + j
                v01 = face_vertex_start + i * (n_segments + 1) + (j + 1)
                v10 = face_vertex_start + (i + 1) * (n_segments + 1) + j
                v11 = face_vertex_start + (i + 1) * (n_segments + 1) + (j + 1)
                
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
    # 正方体每个面的法向量是固定的
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
    HALF_SIZE = 0.5  # 半边长为0.5，则棱长为1
    N_SEGMENTS = 4   # 每个面分成 4x4 的网格
    output = "e:/workspace/NexDynSDF/output/cube.nsm"
    generate_cube_nsm(HALF_SIZE, N_SEGMENTS, output)
