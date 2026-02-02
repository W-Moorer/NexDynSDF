#ifndef NAGATA_TRIANGLE_MESH_DISTANCE_H
#define NAGATA_TRIANGLE_MESH_DISTANCE_H

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>

#include "utils/NagataPatch.h"

namespace sdflib
{
namespace NagataTriangleMeshDistance
{
    /**
     * @brief 最近实体类型枚举
     */
    enum class NearestEntity { V0, V1, V2, E01, E12, E02, F };
    
    /**
     * @brief 距离查询结果结构体
     */
    struct Result
    {
        float distance = std::numeric_limits<float>::max();
        glm::vec3 nearestPoint;
        NearestEntity nearestEntity;
        int triangleId = -1;
    };
    
    /**
     * @brief 包围球结构体
     */
    struct BoundingSphere
    {
        glm::vec3 center;
        float radius;
    };
    
    /**
     * @brief BVH节点结构体
     */
    struct Node
    {
        BoundingSphere bvLeft;
        BoundingSphere bvRight;
        int left = -1;  // 如果left == -1，right存储三角形ID
        int right = -1;
    };
    
    /**
     * @brief 三角形结构体
     */
    struct Triangle
    {
        std::array<glm::vec3, 3> vertices;
        int id = -1;
    };
    
    /**
     * @brief Nagata三角形网格距离查询类
     * 
     * 基于BVH加速结构，使用Nagata曲面进行精确距离查询
     */
    class NagataTriangleMeshDistance
    {
    private:
        std::vector<glm::vec3> vertices;
        std::vector<std::array<int, 3>> triangles;
        std::vector<Node> nodes;
        std::vector<NagataPatch::NagataPatchData> nagataPatches;
        std::vector<glm::vec3> pseudonormalsTriangles;
        std::vector<std::array<glm::vec3, 3>> pseudonormalsEdges;
        std::vector<glm::vec3> pseudonormalsVertices;
        BoundingSphere rootBv;
        bool isConstructed = false;
        
        /**
         * @brief 递归构建BVH树
         */
        void buildTree(int nodeId, BoundingSphere& boundingSphere, 
                      std::vector<Triangle>& triangles, int begin, int end);
        
        /**
         * @brief 递归查询最近点
         */
        void query(Result& result, const Node& node, const glm::vec3& point) const;
        
        /**
         * @brief 计算点到Nagata曲面的距离
         */
        float pointToNagataPatchDistance(const glm::vec3& point, 
                                         const NagataPatch::NagataPatchData& patch,
                                         glm::vec3& nearestPoint,
                                         NearestEntity& nearestEntity) const;
        
    public:
        NagataTriangleMeshDistance() = default;
        
        /**
         * @brief 构造函数
         * 
         * @param vertices 顶点数组
         * @param triangles 三角形索引数组
         * @param vertexNormals 顶点法向量数组（可选，如果不提供则自动计算）
         */
        NagataTriangleMeshDistance(const std::vector<glm::vec3>& vertices,
                                   const std::vector<std::array<int, 3>>& triangles,
                                   const std::vector<glm::vec3>& vertexNormals = {});
        
        /**
         * @brief 构造/重建对象
         */
        void construct(const std::vector<glm::vec3>& vertices,
                      const std::vector<std::array<int, 3>>& triangles,
                      const std::vector<glm::vec3>& vertexNormals = {});
        
        /**
         * @brief 计算无符号距离
         * 
         * @param point 查询点
         * @return Result 查询结果
         */
        Result unsignedDistance(const glm::vec3& point) const;
        Result unsignedDistance(const std::array<float, 3>& point) const;
        
        /**
         * @brief 计算有符号距离
         * 
         * @param point 查询点
         * @return Result 查询结果
         */
        Result signedDistance(const glm::vec3& point) const;
        Result signedDistance(const std::array<float, 3>& point) const;
        
        /**
         * @brief 获取Nagata曲面数据
         */
        const std::vector<NagataPatch::NagataPatchData>& getNagataPatches() const 
        { 
            return nagataPatches; 
        }
        
        /**
         * @brief 检查是否已构造
         */
        bool isConstructed() const { return isConstructed; }
    };
    
    // 内联函数定义
    
    inline NagataTriangleMeshDistance::NagataTriangleMeshDistance(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<int, 3>>& triangles,
        const std::vector<glm::vec3>& vertexNormals)
    {
        construct(vertices, triangles, vertexNormals);
    }
    
    inline void NagataTriangleMeshDistance::construct(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<int, 3>>& triangles,
        const std::vector<glm::vec3>& vertexNormals)
    {
        this->vertices = vertices;
        this->triangles = triangles;
        
        // 计算伪法向量
        pseudonormalsTriangles.resize(triangles.size());
        pseudonormalsEdges.resize(triangles.size());
        pseudonormalsVertices.resize(vertices.size(), glm::vec3(0.0f));
        
        // 边数据结构
        std::unordered_map<uint64_t, glm::vec3> edgeNormals;
        std::unordered_map<uint64_t, int> edgesCount;
        const uint64_t nVertices = vertices.size();
        
        auto addEdgeNormal = [&](int i, int j, const glm::vec3& triangleNormal)
        {
            const uint64_t key = std::min(i, j) * nVertices + std::max(i, j);
            if (edgeNormals.find(key) == edgeNormals.end()) {
                edgeNormals[key] = triangleNormal;
                edgesCount[key] = 1;
            } else {
                edgeNormals[key] += triangleNormal;
                edgesCount[key] += 1;
            }
        };
        
        auto getEdgeNormal = [&](int i, int j) -> glm::vec3
        {
            const uint64_t key = std::min(i, j) * nVertices + std::max(i, j);
            auto it = edgeNormals.find(key);
            if (it != edgeNormals.end()) return it->second;
            return glm::vec3(0.0f);
        };
        
        // 计算每个三角形的法向量和伪法向量
        for (size_t i = 0; i < triangles.size(); i++)
        {
            const auto& tri = triangles[i];
            const glm::vec3& a = vertices[tri[0]];
            const glm::vec3& b = vertices[tri[1]];
            const glm::vec3& c = vertices[tri[2]];
            
            glm::vec3 triangleNormal = glm::normalize(glm::cross(b - a, c - a));
            pseudonormalsTriangles[i] = triangleNormal;
            
            // 顶点伪法向量（角度加权）
            float alpha0 = std::acos(glm::clamp(glm::dot(glm::normalize(b - a), glm::normalize(c - a)), -1.0f, 1.0f));
            float alpha1 = std::acos(glm::clamp(glm::dot(glm::normalize(a - b), glm::normalize(c - b)), -1.0f, 1.0f));
            float alpha2 = std::acos(glm::clamp(glm::dot(glm::normalize(b - c), glm::normalize(a - c)), -1.0f, 1.0f));
            
            pseudonormalsVertices[tri[0]] += alpha0 * triangleNormal;
            pseudonormalsVertices[tri[1]] += alpha1 * triangleNormal;
            pseudonormalsVertices[tri[2]] += alpha2 * triangleNormal;
            
            // 边法向量
            addEdgeNormal(tri[0], tri[1], triangleNormal);
            addEdgeNormal(tri[1], tri[2], triangleNormal);
            addEdgeNormal(tri[0], tri[2], triangleNormal);
        }
        
        // 归一化顶点法向量
        for (auto& n : pseudonormalsVertices)
        {
            n = glm::normalize(n);
            if (std::isnan(n.x) || std::isnan(n.y) || std::isnan(n.z))
                n = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        
        // 归一化边法向量
        for (size_t triI = 0; triI < triangles.size(); triI++)
        {
            const auto& tri = triangles[triI];
            pseudonormalsEdges[triI][0] = glm::normalize(getEdgeNormal(tri[0], tri[1]));
            pseudonormalsEdges[triI][1] = glm::normalize(getEdgeNormal(tri[1], tri[2]));
            pseudonormalsEdges[triI][2] = glm::normalize(getEdgeNormal(tri[0], tri[2]));
            
            for (int k = 0; k < 3; k++)
            {
                if (std::isnan(pseudonormalsEdges[triI][k].x) || 
                    std::isnan(pseudonormalsEdges[triI][k].y) || 
                    std::isnan(pseudonormalsEdges[triI][k].z))
                    pseudonormalsEdges[triI][k] = pseudonormalsTriangles[triI];
            }
        }
        
        // 构建Nagata曲面数据
        nagataPatches.reserve(triangles.size());
        for (size_t i = 0; i < triangles.size(); i++)
        {
            const auto& tri = triangles[i];
            glm::vec3 v0 = vertices[tri[0]];
            glm::vec3 v1 = vertices[tri[1]];
            glm::vec3 v2 = vertices[tri[2]];
            
            glm::vec3 n0, n1, n2;
            if (vertexNormals.empty())
            {
                // 使用计算的伪法向量
                n0 = pseudonormalsVertices[tri[0]];
                n1 = pseudonormalsVertices[tri[1]];
                n2 = pseudonormalsVertices[tri[2]];
            }
            else
            {
                // 使用提供的法向量
                n0 = vertexNormals[tri[0]];
                n1 = vertexNormals[tri[1]];
                n2 = vertexNormals[tri[2]];
            }
            
            nagataPatches.emplace_back(v0, v1, v2, n0, n1, n2);
        }
        
        // 构建BVH树
        std::vector<Triangle> triData;
        triData.reserve(triangles.size());
        for (size_t i = 0; i < triangles.size(); i++)
        {
            Triangle t;
            t.id = static_cast<int>(i);
            t.vertices[0] = vertices[triangles[i][0]];
            t.vertices[1] = vertices[triangles[i][1]];
            t.vertices[2] = vertices[triangles[i][2]];
            triData.push_back(t);
        }
        
        nodes.push_back(Node());
        buildTree(0, rootBv, triData, 0, static_cast<int>(triData.size()));
        
        isConstructed = true;
    }
    
    inline void NagataTriangleMeshDistance::buildTree(
        int nodeId, BoundingSphere& boundingSphere, 
        std::vector<Triangle>& triangles, int begin, int end)
    {
        int nTriangles = end - begin;
        
        if (nTriangles == 0)
        {
            std::cerr << "NagataTriangleMeshDistance::buildTree error: Empty leaf." << std::endl;
            return;
        }
        else if (nTriangles == 1)
        {
            // 叶节点
            nodes[nodeId].left = -1;
            nodes[nodeId].right = triangles[begin].id;
            
            // 计算包围球
            const Triangle& tri = triangles[begin];
            glm::vec3 center = (tri.vertices[0] + tri.vertices[1] + tri.vertices[2]) / 3.0f;
            float radius = std::max({
                glm::length(tri.vertices[0] - center),
                glm::length(tri.vertices[1] - center),
                glm::length(tri.vertices[2] - center)
            });
            
            boundingSphere.center = center;
            boundingSphere.radius = radius;
        }
        else
        {
            // 计算AABB中心和最大维度
            glm::vec3 top(std::numeric_limits<float>::lowest());
            glm::vec3 bottom(std::numeric_limits<float>::max());
            glm::vec3 center(0.0f);
            
            for (int triI = begin; triI < end; triI++)
            {
                for (int vertexI = 0; vertexI < 3; vertexI++)
                {
                    const glm::vec3& p = triangles[triI].vertices[vertexI];
                    center += p;
                    
                    for (int coordI = 0; coordI < 3; coordI++)
                    {
                        top[coordI] = std::max(top[coordI], p[coordI]);
                        bottom[coordI] = std::min(bottom[coordI], p[coordI]);
                    }
                }
            }
            
            center /= static_cast<float>(3 * nTriangles);
            glm::vec3 diagonal = top - bottom;
            int splitDim = 0;
            if (diagonal.y > diagonal.x) splitDim = 1;
            if (diagonal.z > diagonal[splitDim]) splitDim = 2;
            
            // 计算包围球
            float radiusSq = 0.0f;
            for (int triI = begin; triI < end; triI++)
            {
                for (int i = 0; i < 3; i++)
                {
                    float distSq = glm::dot(center - triangles[triI].vertices[i], 
                                           center - triangles[triI].vertices[i]);
                    radiusSq = std::max(radiusSq, distSq);
                }
            }
            
            boundingSphere.center = center;
            boundingSphere.radius = std::sqrt(radiusSq);
            
            // 按中心点排序
            std::sort(triangles.begin() + begin, triangles.begin() + end,
                [splitDim](const Triangle& a, const Triangle& b)
                {
                    return a.vertices[0][splitDim] < b.vertices[0][splitDim];
                }
            );
            
            // 递归构建子节点
            int mid = static_cast<int>(0.5f * (begin + end));
            
            nodes[nodeId].left = static_cast<int>(nodes.size());
            nodes.push_back(Node());
            buildTree(nodes[nodeId].left, nodes[nodeId].bvLeft, triangles, begin, mid);
            
            nodes[nodeId].right = static_cast<int>(nodes.size());
            nodes.push_back(Node());
            buildTree(nodes[nodeId].right, nodes[nodeId].bvRight, triangles, mid, end);
        }
    }
    
    inline void NagataTriangleMeshDistance::query(
        Result& result, const Node& node, const glm::vec3& point) const
    {
        if (node.left == -1)
        {
            // 叶节点：计算到Nagata曲面的距离
            int triangleId = node.right;
            glm::vec3 nearestPoint;
            NearestEntity nearestEntity;
            
            float dist = pointToNagataPatchDistance(point, nagataPatches[triangleId], 
                                                    nearestPoint, nearestEntity);
            
            if (dist < result.distance)
            {
                result.distance = dist;
                result.nearestPoint = nearestPoint;
                result.nearestEntity = nearestEntity;
                result.triangleId = triangleId;
            }
        }
        else
        {
            // 递归查询
            float dLeft = glm::length(point - node.bvLeft.center) - node.bvLeft.radius;
            float dRight = glm::length(point - node.bvRight.center) - node.bvRight.radius;
            
            if (dLeft < dRight)
            {
                if (dLeft < result.distance)
                    query(result, nodes[node.left], point);
                if (dRight < result.distance)
                    query(result, nodes[node.right], point);
            }
            else
            {
                if (dRight < result.distance)
                    query(result, nodes[node.right], point);
                if (dLeft < result.distance)
                    query(result, nodes[node.left], point);
            }
        }
    }
    
    inline float NagataTriangleMeshDistance::pointToNagataPatchDistance(
        const glm::vec3& point, 
        const NagataPatch::NagataPatchData& patch,
        glm::vec3& nearestPoint,
        NearestEntity& nearestEntity) const
    {
        float u, v;
        float distSq = NagataPatch::findNearestPointOnNagataPatch(
            point, patch, nearestPoint, u, v, 10);
        
        // 根据参数确定最近实体类型
        const float eps = 1e-4f;
        if (u < eps && v < eps)
            nearestEntity = NearestEntity::V0;
        else if (u > 1.0f - eps && v < eps)
            nearestEntity = NearestEntity::V1;
        else if (u > 1.0f - eps && v > 1.0f - eps)
            nearestEntity = NearestEntity::V2;
        else if (v < eps)
            nearestEntity = NearestEntity::E01;
        else if (u - v < eps)
            nearestEntity = NearestEntity::E12;
        else if (std::abs(u - 1.0f) < eps && v > eps && v < 1.0f - eps)
            nearestEntity = NearestEntity::E02;
        else
            nearestEntity = NearestEntity::F;
        
        return std::sqrt(distSq);
    }
    
    inline Result NagataTriangleMeshDistance::unsignedDistance(const glm::vec3& point) const
    {
        if (!isConstructed)
        {
            std::cerr << "NagataTriangleMeshDistance error: not constructed." << std::endl;
            return Result();
        }
        
        Result result;
        result.distance = std::numeric_limits<float>::max();
        query(result, nodes[0], point);
        return result;
    }
    
    inline Result NagataTriangleMeshDistance::unsignedDistance(const std::array<float, 3>& point) const
    {
        return unsignedDistance(glm::vec3(point[0], point[1], point[2]));
    }
    
    inline Result NagataTriangleMeshDistance::signedDistance(const glm::vec3& point) const
    {
        Result result = unsignedDistance(point);
        
        if (result.triangleId < 0) return result;
        
        const auto& tri = triangles[result.triangleId];
        glm::vec3 pseudonormal;
        
        switch (result.nearestEntity)
        {
            case NearestEntity::V0:
                pseudonormal = pseudonormalsVertices[tri[0]];
                break;
            case NearestEntity::V1:
                pseudonormal = pseudonormalsVertices[tri[1]];
                break;
            case NearestEntity::V2:
                pseudonormal = pseudonormalsVertices[tri[2]];
                break;
            case NearestEntity::E01:
                pseudonormal = pseudonormalsEdges[result.triangleId][0];
                break;
            case NearestEntity::E12:
                pseudonormal = pseudonormalsEdges[result.triangleId][1];
                break;
            case NearestEntity::E02:
                pseudonormal = pseudonormalsEdges[result.triangleId][2];
                break;
            case NearestEntity::F:
                pseudonormal = pseudonormalsTriangles[result.triangleId];
                break;
        }
        
        glm::vec3 diff = point - result.nearestPoint;
        float sign = glm::dot(diff, pseudonormal) >= 0.0f ? 1.0f : -1.0f;
        result.distance *= sign;
        
        return result;
    }
    
    inline Result NagataTriangleMeshDistance::signedDistance(const std::array<float, 3>& point) const
    {
        return signedDistance(glm::vec3(point[0], point[1], point[2]));
    }
}
}

#endif // NAGATA_TRIANGLE_MESH_DISTANCE_H
