/**
 * @file test_mesh_binary_loader.cpp
 * @brief 测试从二进制文件加载网格数据并使用Nagata曲面
 * 
 * 这个示例展示了如何：
 * 1. 从二进制文件加载网格数据
 * 2. 创建Nagata曲面数据
 * 3. 使用Nagata曲面进行距离查询
 */

#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "SdfLib/utils/MeshBinaryLoader.h"
#include "SdfLib/utils/NagataPatch.h"
#include "SdfLib/NagataTriangleMeshDistance.h"

using namespace sdflib;

int main(int argc, char* argv[])
{
    // 检查命令行参数
    if (argc < 2)
    {
        std::cout << "用法: " << argv[0] << " <mesh_binary_file>" << std::endl;
        std::cout << "示例: " << argv[0] << " ../parsed_meshes/combinatorial_geometry/CompositeBody1_surface_cellnormals.bin" << std::endl;
        return 1;
    }
    
    std::string binaryFile = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "网格二进制文件加载测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 1. 从二进制文件加载网格数据
    std::cout << "\n[1] 从二进制文件加载网格数据..." << std::endl;
    MeshBinaryLoader::MeshData meshData = MeshBinaryLoader::loadFromBinary(binaryFile);
    
    if (meshData.getNumVertices() == 0 || meshData.getNumFaces() == 0)
    {
        std::cerr << "错误: 无法加载网格数据" << std::endl;
        return 1;
    }
    
    // 计算边界框
    glm::vec3 minBound, maxBound;
    meshData.computeBounds(minBound, maxBound);
    std::cout << "边界框: [" << minBound.x << ", " << minBound.y << ", " << minBound.z << "] ~ ["
              << maxBound.x << ", " << maxBound.y << ", " << maxBound.z << "]" << std::endl;
    
    // 2. 创建Nagata曲面数据
    std::cout << "\n[2] 创建Nagata曲面数据..." << std::endl;
    std::vector<NagataPatch::NagataPatchData> nagataPatches = 
        MeshBinaryLoader::createNagataPatchData(meshData);
    std::cout << "成功创建 " << nagataPatches.size() << " 个Nagata曲面" << std::endl;
    
    // 3. 创建Nagata距离查询对象
    std::cout << "\n[3] 创建Nagata距离查询对象..." << std::endl;
    
    // 准备顶点数组
    std::vector<glm::vec3> vertices = meshData.vertices;
    std::vector<std::array<int, 3>> triangles;
    triangles.reserve(meshData.faces.size());
    for (const auto& face : meshData.faces)
    {
        triangles.push_back({static_cast<int>(face[0]), 
                            static_cast<int>(face[1]), 
                            static_cast<int>(face[2])});
    }
    
    // 准备顶点法向量（使用面片法向量的平均值）
    std::vector<glm::vec3> vertexNormals(meshData.getNumVertices(), glm::vec3(0.0f));
    std::vector<int> vertexFaceCount(meshData.getNumVertices(), 0);
    
    for (size_t i = 0; i < meshData.getNumFaces(); ++i)
    {
        auto faceNormals = meshData.getFaceVertexNormals(i);
        for (int j = 0; j < 3; ++j)
        {
            uint32_t vertexIdx = meshData.faces[i][j];
            vertexNormals[vertexIdx] += faceNormals[j];
            vertexFaceCount[vertexIdx]++;
        }
    }
    
    for (size_t i = 0; i < vertexNormals.size(); ++i)
    {
        if (vertexFaceCount[i] > 0)
        {
            vertexNormals[i] /= static_cast<float>(vertexFaceCount[i]);
            vertexNormals[i] = glm::normalize(vertexNormals[i]);
        }
    }
    
    // 创建Nagata距离查询对象
    NagataTriangleMeshDistance::NagataTriangleMeshDistance meshDist(
        vertices, triangles, vertexNormals);
    
    if (!meshDist.isConstructed())
    {
        std::cerr << "错误: 无法构建Nagata距离查询对象" << std::endl;
        return 1;
    }
    std::cout << "Nagata距离查询对象构建成功" << std::endl;
    
    // 4. 测试距离查询
    std::cout << "\n[4] 测试距离查询..." << std::endl;
    
    // 在边界框中心查询
    glm::vec3 center = (minBound + maxBound) * 0.5f;
    auto result = meshDist.signedDistance(center);
    std::cout << "查询点 (中心): [" << center.x << ", " << center.y << ", " << center.z << "]" << std::endl;
    std::cout << "  有符号距离: " << result.distance << std::endl;
    std::cout << "  最近点: [" << result.nearestPoint.x << ", " 
              << result.nearestPoint.y << ", " << result.nearestPoint.z << "]" << std::endl;
    std::cout << "  三角形ID: " << result.triangleId << std::endl;
    
    // 在边界框外查询
    glm::vec3 outside = maxBound + glm::vec3(10.0f, 10.0f, 10.0f);
    result = meshDist.signedDistance(outside);
    std::cout << "\n查询点 (外部): [" << outside.x << ", " << outside.y << ", " << outside.z << "]" << std::endl;
    std::cout << "  有符号距离: " << result.distance << std::endl;
    
    // 5. 测试Nagata曲面求值
    std::cout << "\n[5] 测试Nagata曲面求值..." << std::endl;
    if (!nagataPatches.empty())
    {
        const auto& patch = nagataPatches[0];
        
        // 在面片重心处求值
        float u = 0.5f;
        float v = 0.25f;
        glm::vec3 surfacePoint = NagataPatch::evaluateSurface(patch, u, v);
        glm::vec3 normal = NagataPatch::evaluateNormal(patch, u, v);
        
        std::cout << "面片0在 (u=" << u << ", v=" << v << ") 处:" << std::endl;
        std::cout << "  曲面点: [" << surfacePoint.x << ", " << surfacePoint.y << ", " << surfacePoint.z << "]" << std::endl;
        std::cout << "  法向量: [" << normal.x << ", " << normal.y << ", " << normal.z << "]" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试完成" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
