"""
将cone_two_faces.nsm中的非水平面绕共享边旋转90度，生成钝角模型
"""
import numpy as np
import struct
import os


def load_nsm_simple(filepath):
    """
    读取NSM文件
    
    Args:
        filepath: NSM文件路径
        
    Returns:
        vertices, triangles, tri_face_ids, tri_vertex_normals
    """
    with open(filepath, 'rb') as f:
        header_data = f.read(64)
        num_vertices = struct.unpack('<I', header_data[8:12])[0]
        num_triangles = struct.unpack('<I', header_data[12:16])[0]
        
        vertices = np.fromfile(f, dtype=np.float64, count=num_vertices * 3).reshape(num_vertices, 3)
        triangles = np.fromfile(f, dtype=np.uint32, count=num_triangles * 3).reshape(num_triangles, 3)
        tri_face_ids = np.fromfile(f, dtype=np.uint32, count=num_triangles)
        tri_vertex_normals = np.fromfile(f, dtype=np.float64, count=num_triangles * 3 * 3).reshape(num_triangles, 3, 3)
        
        return vertices, triangles, tri_face_ids, tri_vertex_normals


def save_nsm(filepath, vertices, triangles, tri_face_ids, tri_vertex_normals):
    """
    保存NSM文件
    
    Args:
        filepath: 输出文件路径
        vertices: 顶点坐标
        triangles: 三角形索引
        tri_face_ids: 面片ID
        tri_vertex_normals: 顶点法向量
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(b'NSM\x00')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(vertices)))
        f.write(struct.pack('<I', len(triangles)))
        f.write(b'\x00' * 48)
        vertices.tofile(f)
        triangles.tofile(f)
        tri_face_ids.tofile(f)
        tri_vertex_normals.tofile(f)
    print(f"已保存NSM文件: {filepath}")


def compute_triangle_normal(v0, v1, v2):
    """
    计算三角形法向量
    
    Args:
        v0, v1, v2: 三角形三个顶点
        
    Returns:
        归一化的法向量
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal = normal / norm
    return normal


def rotation_matrix_axis_angle(axis, angle):
    """
    根据旋转轴和角度创建旋转矩阵（Rodrigues公式）
    
    Args:
        axis: 旋转轴（单位向量）
        angle: 旋转角度（弧度）
        
    Returns:
        3x3旋转矩阵
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def create_obtuse_model(input_path, output_path):
    """
    将非水平面绕共享边逆时针旋转90度，创建钝角模型
    
    Args:
        input_path: 输入NSM文件路径
        output_path: 输出NSM文件路径
    """
    print(f"读取文件: {input_path}")
    vertices, triangles, tri_face_ids, tri_vertex_normals = load_nsm_simple(input_path)
    
    print(f"顶点数: {len(vertices)}, 三角形数: {len(triangles)}")
    print(f"面片ID: {tri_face_ids}")
    
    # 找出两个三角形
    tri0 = triangles[0]
    tri1 = triangles[1]
    
    v0_tri0, v1_tri0, v2_tri0 = vertices[tri0]
    v0_tri1, v1_tri1, v2_tri1 = vertices[tri1]
    
    # 计算两个三角形的法向量
    normal0 = compute_triangle_normal(v0_tri0, v1_tri0, v2_tri0)
    normal1 = compute_triangle_normal(v0_tri1, v1_tri1, v2_tri1)
    
    print(f"\n三角形0法向量: {normal0}")
    print(f"三角形1法向量: {normal1}")
    
    # 判断哪个是水平面（法向量接近Z轴）
    z_axis = np.array([0, 0, 1])
    dot0 = abs(np.dot(normal0, z_axis))
    dot1 = abs(np.dot(normal1, z_axis))
    
    if dot0 > dot1:
        horizontal_tri_idx = 0
        non_horizontal_tri_idx = 1
        print(f"\n三角形0是水平面 (与Z轴夹角: {np.degrees(np.arccos(dot0)):.2f}°)")
        print(f"三角形1是非水平面 (与Z轴夹角: {np.degrees(np.arccos(dot1)):.2f}°)")
    else:
        horizontal_tri_idx = 1
        non_horizontal_tri_idx = 0
        print(f"\n三角形1是水平面 (与Z轴夹角: {np.degrees(np.arccos(dot1)):.2f}°)")
        print(f"三角形0是非水平面 (与Z轴夹角: {np.degrees(np.arccos(dot0)):.2f}°)")
    
    # 找到共享边
    set0 = set(triangles[horizontal_tri_idx])
    set1 = set(triangles[non_horizontal_tri_idx])
    shared_vertices = list(set0.intersection(set1))
    
    if len(shared_vertices) != 2:
        print("错误: 未找到共享边")
        return
    
    print(f"\n共享边顶点索引: {shared_vertices}")
    print(f"共享边顶点坐标: {vertices[shared_vertices[0]]}, {vertices[shared_vertices[1]]}")
    
    # 找到非水平面上不共享的顶点
    non_horizontal_tri = triangles[non_horizontal_tri_idx]
    non_shared_vertex_idx = None
    for v_idx in non_horizontal_tri:
        if v_idx not in shared_vertices:
            non_shared_vertex_idx = v_idx
            break
    
    print(f"非水平面上不共享的顶点索引: {non_shared_vertex_idx}")
    print(f"该顶点原始坐标: {vertices[non_shared_vertex_idx]}")
    
    # 计算共享边向量作为旋转轴
    edge_start = vertices[shared_vertices[0]]
    edge_end = vertices[shared_vertices[1]]
    rotation_axis = edge_end - edge_start
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    print(f"\n旋转轴（共享边方向）: {rotation_axis}")
    
    # 计算90度逆时针旋转矩阵
    # 从共享边的一个端点看，逆时针旋转
    angle = np.pi / 2  # 90度
    R = rotation_matrix_axis_angle(rotation_axis, angle)
    
    print(f"旋转角度: 90度 (逆时针)")
    
    # 旋转非共享顶点（绕共享边的中点旋转）
    edge_center = (edge_start + edge_end) / 2
    print(f"旋转中心（共享边中点）: {edge_center}")
    
    # 创建新的顶点数组（深拷贝）
    new_vertices = vertices.copy()
    
    # 旋转非共享顶点
    old_pos = vertices[non_shared_vertex_idx]
    # 平移到原点，旋转，再平移回去
    relative_pos = old_pos - edge_center
    new_relative_pos = R @ relative_pos
    new_pos = new_relative_pos + edge_center
    new_vertices[non_shared_vertex_idx] = new_pos
    
    print(f"\n顶点 {non_shared_vertex_idx} 旋转后坐标: {new_pos}")
    
    # 更新非水平面的法向量：旋转每个顶点的法向量
    new_tri_vertex_normals = tri_vertex_normals.copy()
    
    # 对于非水平面的三角形，旋转其每个顶点的法向量
    for j in range(3):
        old_normal = tri_vertex_normals[non_horizontal_tri_idx, j]
        new_normal = R @ old_normal
        # 确保法向量归一化
        new_normal = new_normal / np.linalg.norm(new_normal)
        new_tri_vertex_normals[non_horizontal_tri_idx, j] = new_normal
        print(f"  顶点{j}法向量: {old_normal} -> {new_normal}")
    
    # 计算新的三角形几何法向量（用于计算二面角）
    new_tri = triangles[non_horizontal_tri_idx]
    new_v0 = new_vertices[new_tri[0]]
    new_v1 = new_vertices[new_tri[1]]
    new_v2 = new_vertices[new_tri[2]]
    new_geom_normal = compute_triangle_normal(new_v0, new_v1, new_v2)
    
    print(f"\n非水平面新几何法向量: {new_geom_normal}")
    
    # 计算新的夹角
    horizontal_normal = compute_triangle_normal(
        vertices[triangles[horizontal_tri_idx, 0]],
        vertices[triangles[horizontal_tri_idx, 1]],
        vertices[triangles[horizontal_tri_idx, 2]]
    )
    
    # 两个法向量之间的夹角
    angle_between_normals = np.arccos(np.clip(np.dot(new_geom_normal, horizontal_normal), -1, 1))
    dihedral_angle = np.pi - angle_between_normals
    
    print(f"\n旋转后两面夹角（二面角）: {np.degrees(dihedral_angle):.2f}°")
    
    # 保存新文件
    save_nsm(output_path, new_vertices, triangles, tri_face_ids, new_tri_vertex_normals)
    
    print(f"\n完成！钝角模型已保存至: {output_path}")


if __name__ == "__main__":
    input_file = "e:/workspace/NexDynSDF/output/cone_two_faces.nsm"
    output_file = "e:/workspace/NexDynSDF/output/cone_two_faces_obtuse.nsm"
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
    else:
        create_obtuse_model(input_file, output_file)
