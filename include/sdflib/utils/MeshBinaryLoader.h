#ifndef MESH_BINARY_LOADER_H
#define MESH_BINARY_LOADER_H

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <cstdint>
#include "NagataPatch.h"

namespace sdflib
{
namespace MeshBinaryLoader
{
    /**
     * @brief 网格数据结构
     * 
     * 存储从二进制文件加载的网格数据
     */
    struct MeshData
    {
        std::vector<glm::vec3> vertices;                    // 顶点坐标数组
        std::vector<std::array<uint32_t, 3>> faces;        // 面片索引数组
        std::vector<std::array<glm::vec3, 3>> faceNormals; // 面片顶点法向量数组
        
        /**
         * @brief 获取顶点数量
         */
        size_t getNumVertices() const { return vertices.size(); }
        
        /**
         * @brief 获取面片数量
         */
        size_t getNumFaces() const { return faces.size(); }
        
        /**
         * @brief 获取指定面片的顶点坐标
         * 
         * @param faceIndex 面片索引
         * @return std::array<glm::vec3, 3> 三个顶点的坐标
         */
        std::array<glm::vec3, 3> getFaceVertices(size_t faceIndex) const
        {
            std::array<glm::vec3, 3> result;
            for (int i = 0; i < 3; ++i)
            {
                result[i] = vertices[faces[faceIndex][i]];
            }
            return result;
        }
        
        /**
         * @brief 获取指定面片的顶点法向量
         * 
         * @param faceIndex 面片索引
         * @return std::array<glm::vec3, 3> 三个顶点在该面片内的法向量
         */
        std::array<glm::vec3, 3> getFaceVertexNormals(size_t faceIndex) const
        {
            return faceNormals[faceIndex];
        }
        
        /**
         * @brief 计算边界框
         * 
         * @param minBound 输出：最小边界
         * @param maxBound 输出：最大边界
         */
        void computeBounds(glm::vec3& minBound, glm::vec3& maxBound) const
        {
            if (vertices.empty())
            {
                minBound = glm::vec3(0.0f);
                maxBound = glm::vec3(0.0f);
                return;
            }
            
            minBound = vertices[0];
            maxBound = vertices[0];
            
            for (const auto& v : vertices)
            {
                minBound = glm::min(minBound, v);
                maxBound = glm::max(maxBound, v);
            }
        }
    };
    
    /**
     * @brief 从二进制文件加载网格数据
     * 
     * 文件格式:
     *     - 4字节: 顶点数量 (uint32)
     *     - 4字节: 面片数量 (uint32)
     *     - N*12字节: 顶点坐标数组 (float32 * 3 * N)
     *     - M*12字节: 面片索引数组 (uint32 * 3 * M)
     *     - M*36字节: 面片顶点法向量数组 (float32 * 3 * 3 * M)
     * 
     * @param filepath 二进制文件路径
     * @return MeshData 加载的网格数据
     */
    inline MeshData loadFromBinary(const std::string& filepath)
    {
        MeshData meshData;
        
        // 打开文件
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "MeshBinaryLoader: 无法打开文件 " << filepath << std::endl;
            return meshData;
        }
        
        // 读取头部信息
        uint32_t numVertices = 0;
        uint32_t numFaces = 0;
        
        file.read(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&numFaces), sizeof(uint32_t));
        
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: 读取头部信息失败 " << filepath << std::endl;
            return meshData;
        }
        
        // 分配内存
        meshData.vertices.resize(numVertices);
        meshData.faces.resize(numFaces);
        meshData.faceNormals.resize(numFaces);
        
        // 读取顶点坐标
        for (uint32_t i = 0; i < numVertices; ++i)
        {
            float x, y, z;
            file.read(reinterpret_cast<char*>(&x), sizeof(float));
            file.read(reinterpret_cast<char*>(&y), sizeof(float));
            file.read(reinterpret_cast<char*>(&z), sizeof(float));
            meshData.vertices[i] = glm::vec3(x, y, z);
        }
        
        // 读取面片索引
        for (uint32_t i = 0; i < numFaces; ++i)
        {
            uint32_t i0, i1, i2;
            file.read(reinterpret_cast<char*>(&i0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i1), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i2), sizeof(uint32_t));
            meshData.faces[i] = {i0, i1, i2};
        }
        
        // 读取面片顶点法向量
        for (uint32_t i = 0; i < numFaces; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                float nx, ny, nz;
                file.read(reinterpret_cast<char*>(&nx), sizeof(float));
                file.read(reinterpret_cast<char*>(&ny), sizeof(float));
                file.read(reinterpret_cast<char*>(&nz), sizeof(float));
                meshData.faceNormals[i][j] = glm::vec3(nx, ny, nz);
            }
        }
        
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: 读取数据失败 " << filepath << std::endl;
            meshData.vertices.clear();
            meshData.faces.clear();
            meshData.faceNormals.clear();
            return meshData;
        }
        
        file.close();
        
        std::cout << "MeshBinaryLoader: 成功加载 " << filepath << std::endl;
        std::cout << "  顶点数量: " << numVertices << std::endl;
        std::cout << "  面片数量: " << numFaces << std::endl;
        
        return meshData;
    }
    
    /**
     * @brief 从二进制文件创建NagataPatchData数组
     * 
     * 使用加载的网格数据创建Nagata曲面数据，用于Nagata插值
     * 
     * @param meshData 网格数据
     * @return std::vector<NagataPatch::NagataPatchData> Nagata曲面数据数组
     */
    inline std::vector<NagataPatch::NagataPatchData> createNagataPatchData(
        const MeshData& meshData)
    {
        std::vector<NagataPatch::NagataPatchData> nagataPatches;
        nagataPatches.reserve(meshData.getNumFaces());
        
        for (size_t i = 0; i < meshData.getNumFaces(); ++i)
        {
            // 获取面片顶点坐标
            auto faceVertices = meshData.getFaceVertices(i);
            
            // 获取面片顶点法向量
            auto faceNormals = meshData.getFaceVertexNormals(i);
            
            // 创建Nagata曲面数据
            nagataPatches.emplace_back(
                faceVertices[0], faceVertices[1], faceVertices[2],
                faceNormals[0], faceNormals[1], faceNormals[2]
            );
        }
        
        return nagataPatches;
    }
}
}

#endif // MESH_BINARY_LOADER_H
