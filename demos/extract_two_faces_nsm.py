import numpy as np
import struct
import os
import sys

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

def extract_two_faces(input_path, output_path):
    print(f"正在从 {input_path} 寻找一对相邻面片...")
    vertices, triangles, tri_face_ids, tri_vertex_normals = load_nsm_simple(input_path)
    
    # 1. 分离侧面和底面三角形
    side_indices = np.where(tri_face_ids == 0)[0]
    base_indices = np.where(tri_face_ids == 1)[0]
    
    # 2. 寻找共享两个顶点（一条边）的一对面片
    found_side_idx = -1
    found_base_idx = -1
    
    for s_idx in side_indices:
        s_tri = set(triangles[s_idx])
        for b_idx in base_indices:
            b_tri = set(triangles[b_idx])
            common = s_tri.intersection(b_tri)
            if len(common) == 2: # 共享一条边
                found_side_idx = s_idx
                found_base_idx = b_idx
                break
        if found_side_idx != -1:
            break
            
    if found_side_idx == -1:
        print("错误: 未找到共享边缘的一对面片。")
        return
        
    print(f"找到共享边缘的面片对: 侧面三角形索引 {found_side_idx}, 底面三角形索引 {found_base_idx}")
    
    # 3. 提取数据
    extracted_indices = [found_side_idx, found_base_idx]
    new_triangles_list = triangles[extracted_indices]
    new_face_ids = tri_face_ids[extracted_indices]
    new_vertex_normals = tri_vertex_normals[extracted_indices]
    
    unique_v_ids = np.unique(new_triangles_list)
    v_map = {old_id: new_id for new_id, old_id in enumerate(unique_v_ids)}
    new_vertices = vertices[unique_v_ids]
    remapped_triangles = np.vectorize(v_map.get)(new_triangles_list).astype(np.uint32)
    
    # 4. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(b'NSM\x00')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(new_vertices)))
        f.write(struct.pack('<I', 2))
        f.write(b'\x00' * 48)
        new_vertices.tofile(f)
        remapped_triangles.tofile(f)
        new_face_ids.tofile(f)
        new_vertex_normals.tofile(f)
        
    print(f"成功保存一对相邻面片至: {output_path}")
    print(f"最终结果: {len(new_vertices)} 顶点, 2 三角形")

if __name__ == "__main__":
    input_file = "e:/workspace/NexDynSDF/output/cone_junction.nsm"
    output_file = "e:/workspace/NexDynSDF/output/cone_two_faces.nsm"
    
    if not os.path.exists(input_file):
        # 兜底：如果 junction 没生成，尝试从原始模型提取
        input_file = "e:/workspace/NexDynSDF/output/cone.nsm"
        
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)
        
    extract_two_faces(input_file, output_file)
