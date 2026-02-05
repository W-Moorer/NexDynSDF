import numpy as np
import struct
import os
import sys

# 尝试导入 load_nsm, 如果不行就直接定义读取逻辑
def load_nsm_simple(filepath):
    with open(filepath, 'rb') as f:
        header_data = f.read(64)
        num_vertices = struct.unpack('<I', header_data[8:12])[0]
        num_triangles = struct.unpack('<I', header_data[12:16])[0]
        
        vertices = np.fromfile(f, dtype=np.float64, count=num_vertices * 3).reshape(num_vertices, 3)
        triangles = np.fromfile(f, dtype=np.uint32, count=num_triangles * 3).reshape(num_triangles, 3)
        tri_face_ids = np.fromfile(f, dtype=np.uint32, count=num_triangles)
        tri_vertex_normals = np.fromfile(f, dtype=np.float64, count=num_triangles * 3 * 3).reshape(num_triangles, 3, 3)
        
        return vertices, triangles, tri_face_ids, tri_vertex_normals

def extract_junction_triangles(input_path, output_path):
    print(f"正在从 {input_path} 提取交界处面片...")
    vertices, triangles, tri_face_ids, tri_vertex_normals = load_nsm_simple(input_path)
    
    # 1. 识别侧面和底面共享的顶点 (Junction Vertices)
    # Face ID 0: 侧面, Face ID 1: 底面
    side_indices = np.where(tri_face_ids == 0)[0]
    base_indices = np.where(tri_face_ids == 1)[0]
    
    side_vertices_ids = set()
    for idx in side_indices:
        side_vertices_ids.update(triangles[idx])
        
    base_vertices_ids = set()
    for idx in base_indices:
        base_vertices_ids.update(triangles[idx])
        
    junction_vertices_ids = side_vertices_ids.intersection(base_vertices_ids)
    print(f"找到 {len(junction_vertices_ids)} 个交界处顶点。")
    
    # 2. 提取所有包含至少一个交界顶点的三角形
    extracted_tri_indices = []
    for i in range(len(triangles)):
        tri = triangles[i]
        if any(v_id in junction_vertices_ids for v_id in tri):
            extracted_tri_indices.append(i)
            
    extracted_tri_indices = np.array(extracted_tri_indices)
    print(f"提取了 {len(extracted_tri_indices)} 个交界处相关的三角形。")
    
    # 3. 准备提取出的数据，并重新映射顶点索引
    new_triangles_list = triangles[extracted_tri_indices]
    new_face_ids = tri_face_ids[extracted_tri_indices]
    new_vertex_normals = tri_vertex_normals[extracted_tri_indices]
    
    # 获取涉及到的所有唯一顶点
    unique_v_ids = np.unique(new_triangles_list)
    v_map = {old_id: new_id for new_id, old_id in enumerate(unique_v_ids)}
    
    new_vertices = vertices[unique_v_ids]
    
    # 重映射三角形索引
    remapped_triangles = np.vectorize(v_map.get)(new_triangles_list).astype(np.uint32)
    
    # 4. 保存为NSM文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(b'NSM\x00')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(new_vertices)))
        f.write(struct.pack('<I', len(remapped_triangles)))
        f.write(b'\x00' * 48)
        new_vertices.tofile(f)
        remapped_triangles.tofile(f)
        new_face_ids.tofile(f)
        new_vertex_normals.tofile(f)
        
    print(f"提取完成，保存文件为: {output_path}")
    print(f"最终结果: {len(new_vertices)} 顶点, {len(remapped_triangles)} 三角形")

if __name__ == "__main__":
    input_file = "e:/workspace/NexDynSDF/output/cone.nsm"
    output_file = "e:/workspace/NexDynSDF/output/cone_junction.nsm"
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)
        
    extract_junction_triangles(input_file, output_file)
