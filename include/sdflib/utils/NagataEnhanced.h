#ifndef NAGATA_ENHANCED_H
#define NAGATA_ENHANCED_H

/**
 * @file NagataEnhanced.h
 * @brief Nagata 增强模块 - 裂隙边修复与 ENG 文件缓存
 * 
 * 功能:
 * - .eng 文件读写 (Enhanced Nagata Geometry)
 * - 裂隙边检测 (Crease Edge Detection)
 * - 共享边界系数计算 (c_sharp)
 * - 自动缓存逻辑
 */

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include "NagataPatch.h"

namespace sdflib
{
namespace NagataEnhanced
{
    // ============================================================
    // 常量定义
    // ============================================================
    
    constexpr char ENG_MAGIC[4] = {'E', 'N', 'G', '\0'};
    constexpr uint32_t ENG_VERSION = 1;
    constexpr float GAP_THRESHOLD = 1e-4f;
    
    // ============================================================
    // 数据结构
    // ============================================================
    
    /**
     * @brief 边键 (排序后的顶点索引对)
     */
    struct EdgeKey
    {
        uint32_t v0;
        uint32_t v1;
        
        EdgeKey() : v0(0), v1(0) {}
        EdgeKey(uint32_t a, uint32_t b) : v0(std::min(a,b)), v1(std::max(a,b)) {}
        
        bool operator<(const EdgeKey& other) const
        {
            if (v0 != other.v0) return v0 < other.v0;
            return v1 < other.v1;
        }
        
        bool operator==(const EdgeKey& other) const
        {
            return v0 == other.v0 && v1 == other.v1;
        }
    };
    
    /**
     * @brief 裂隙边信息
     */
    struct CreaseEdgeInfo
    {
        glm::vec3 A, B;           // 端点坐标
        glm::vec3 n_A_L, n_A_R;   // A 点两侧法向
        glm::vec3 n_B_L, n_B_R;   // B 点两侧法向
        int tri_L, tri_R;         // 相邻三角形索引
        float max_gap;            // 最大间隙
    };
    
    /**
     * @brief 增强 Nagata 数据
     */
    struct EnhancedNagataData
    {
        std::map<EdgeKey, glm::vec3> c_sharps;  // 边 -> c_sharp 系数
        
        bool hasEdge(const EdgeKey& key) const
        {
            return c_sharps.find(key) != c_sharps.end();
        }
        
        glm::vec3 getCSharp(const EdgeKey& key) const
        {
            auto it = c_sharps.find(key);
            if (it != c_sharps.end()) return it->second;
            return glm::vec3(0.0f);
        }
        
        size_t size() const { return c_sharps.size(); }
        bool empty() const { return c_sharps.empty(); }
    };
    
    // ============================================================
    // ENG 文件 I/O
    // ============================================================
    
    /**
     * @brief 从 .eng 文件加载增强数据
     */
    inline bool loadEnhancedData(const std::string& filepath, EnhancedNagataData& data)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }
        
        // 读取头部
        char magic[4];
        file.read(magic, 4);
        if (std::memcmp(magic, ENG_MAGIC, 4) != 0)
        {
            std::cerr << "NagataEnhanced: 无效的 ENG 文件 magic" << std::endl;
            return false;
        }
        
        uint32_t version, numEdges, reserved;
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&numEdges), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&reserved), sizeof(uint32_t));
        
        if (version != ENG_VERSION)
        {
            std::cerr << "NagataEnhanced: 不支持的 ENG 版本 " << version << std::endl;
            return false;
        }
        
        // 读取数据
        data.c_sharps.clear();
        for (uint32_t i = 0; i < numEdges; ++i)
        {
            uint32_t v0, v1;
            float cx, cy, cz;
            
            file.read(reinterpret_cast<char*>(&v0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&v1), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&cx), sizeof(float));
            file.read(reinterpret_cast<char*>(&cy), sizeof(float));
            file.read(reinterpret_cast<char*>(&cz), sizeof(float));
            
            if (!file)
            {
                std::cerr << "NagataEnhanced: ENG 文件数据不完整" << std::endl;
                return false;
            }
            
            data.c_sharps[EdgeKey(v0, v1)] = glm::vec3(cx, cy, cz);
        }
        
        std::cout << "NagataEnhanced: 已加载 " << numEdges << " 条裂隙边数据从 " << filepath << std::endl;
        return true;
    }
    
    /**
     * @brief 保存增强数据到 .eng 文件
     */
    inline bool saveEnhancedData(const std::string& filepath, const EnhancedNagataData& data)
    {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "NagataEnhanced: 无法创建文件 " << filepath << std::endl;
            return false;
        }
        
        // 写入头部
        file.write(ENG_MAGIC, 4);
        uint32_t version = ENG_VERSION;
        uint32_t numEdges = static_cast<uint32_t>(data.c_sharps.size());
        uint32_t reserved = 0;
        
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&numEdges), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&reserved), sizeof(uint32_t));
        
        // 写入数据
        for (const auto& [key, c] : data.c_sharps)
        {
            uint32_t v0 = key.v0;
            uint32_t v1 = key.v1;
            float cx = c.x, cy = c.y, cz = c.z;
            
            file.write(reinterpret_cast<const char*>(&v0), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&v1), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&cx), sizeof(float));
            file.write(reinterpret_cast<const char*>(&cy), sizeof(float));
            file.write(reinterpret_cast<const char*>(&cz), sizeof(float));
        }
        
        std::cout << "NagataEnhanced: 已保存 " << numEdges << " 条裂隙边数据到 " << filepath << std::endl;
        return true;
    }
    
    /**
     * @brief 从 NSM 文件路径派生 ENG 文件路径
     */
    inline std::string getEngFilepath(const std::string& nsmPath)
    {
        size_t dotPos = nsmPath.rfind('.');
        if (dotPos == std::string::npos)
        {
            return nsmPath + ".eng";
        }
        return nsmPath.substr(0, dotPos) + ".eng";
    }
    
    /**
     * @brief 检查 ENG 缓存文件是否存在
     */
    inline bool hasEngCache(const std::string& nsmPath)
    {
        std::ifstream file(getEngFilepath(nsmPath));
        return file.good();
    }
    
    // ============================================================
    // 裂隙边检测
    // ============================================================
    
    /**
     * @brief 检测裂隙边
     * 
     * @param vertices 顶点数组
     * @param faces 面片索引数组 (每个面片3个顶点索引)
     * @param faceNormals 面片顶点法向数组 (每个面片3个法向)
     * @param gapThreshold 间隙阈值
     */
    inline std::map<EdgeKey, CreaseEdgeInfo> detectCreaseEdges(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        float gapThreshold = GAP_THRESHOLD)
    {
        // 边到三角形的映射
        struct EdgeTriInfo
        {
            int triIdx;
            uint32_t v0, v1;
            int local0, local1;
        };
        
        std::map<EdgeKey, std::vector<EdgeTriInfo>> edgeToTris;
        
        // 遍历所有三角形，收集边信息
        for (size_t triIdx = 0; triIdx < faces.size(); ++triIdx)
        {
            const auto& tri = faces[triIdx];
            
            // 三条边: (0,1), (1,2), (0,2)
            std::array<std::tuple<int, int, int, int>, 3> edges = {{
                {0, 1, 0, 1},
                {1, 2, 1, 2},
                {0, 2, 0, 2}
            }};
            
            for (const auto& [l0, l1, local0, local1] : edges)
            {
                EdgeKey key(tri[l0], tri[l1]);
                EdgeTriInfo info;
                info.triIdx = static_cast<int>(triIdx);
                info.v0 = tri[l0];
                info.v1 = tri[l1];
                info.local0 = local0;
                info.local1 = local1;
                edgeToTris[key].push_back(info);
            }
        }
        
        std::map<EdgeKey, CreaseEdgeInfo> creaseEdges;
        
        // 检测裂隙边
        for (const auto& [edgeKey, trisInfo] : edgeToTris)
        {
            if (trisInfo.size() != 2)
                continue;  // 边界边或非流形
            
            const auto& triL = trisInfo[0];
            const auto& triR = trisInfo[1];
            
            glm::vec3 A = vertices[edgeKey.v0];
            glm::vec3 B = vertices[edgeKey.v1];
            
            // 获取两侧法向
            auto getNormalAtVertex = [&](int triIdx, uint32_t globalVIdx) -> glm::vec3
            {
                const auto& tri = faces[triIdx];
                for (int i = 0; i < 3; ++i)
                {
                    if (tri[i] == globalVIdx)
                        return faceNormals[triIdx][i];
                }
                return glm::vec3(0.0f, 0.0f, 1.0f);
            };
            
            glm::vec3 n_A_L = getNormalAtVertex(triL.triIdx, edgeKey.v0);
            glm::vec3 n_B_L = getNormalAtVertex(triL.triIdx, edgeKey.v1);
            glm::vec3 n_A_R = getNormalAtVertex(triR.triIdx, edgeKey.v0);
            glm::vec3 n_B_R = getNormalAtVertex(triR.triIdx, edgeKey.v1);
            
            // 计算两侧边界曲线系数
            glm::vec3 e = B - A;
            glm::vec3 c_L = NagataPatch::computeCurvature(e, n_A_L, n_B_L);
            glm::vec3 c_R = NagataPatch::computeCurvature(e, n_A_R, n_B_R);
            
            // 采样比较
            float maxGap = 0.0f;
            for (int i = 0; i <= 10; ++i)
            {
                float t = static_cast<float>(i) / 10.0f;
                glm::vec3 p_L = (1.0f - t) * A + t * B - c_L * t * (1.0f - t);
                glm::vec3 p_R = (1.0f - t) * A + t * B - c_R * t * (1.0f - t);
                float gap = glm::length(p_L - p_R);
                maxGap = std::max(maxGap, gap);
            }
            
            if (maxGap > gapThreshold)
            {
                CreaseEdgeInfo info;
                info.A = A;
                info.B = B;
                info.n_A_L = n_A_L;
                info.n_A_R = n_A_R;
                info.n_B_L = n_B_L;
                info.n_B_R = n_B_R;
                info.tri_L = triL.triIdx;
                info.tri_R = triR.triIdx;
                info.max_gap = maxGap;
                creaseEdges[edgeKey] = info;
            }
        }
        
        return creaseEdges;
    }
    
    // ============================================================
    // c_sharp 计算
    // ============================================================
    
    /**
     * @brief 计算折痕方向 (两侧切平面的交线)
     */
    inline glm::vec3 computeCreaseDirection(glm::vec3 nL, glm::vec3 nR, glm::vec3 e)
    {
        glm::vec3 d = glm::cross(nL, nR);
        float len = glm::length(d);
        
        if (len < 1e-8f)
        {
            // 退化: 法向几乎平行，使用边方向
            return glm::normalize(e);
        }
        
        d = d / len;
        
        // 确保方向与边方向一致
        if (glm::dot(d, e) < 0.0f)
            d = -d;
        
        return d;
    }
    
    /**
     * @brief 计算共享边界系数 c_sharp
     * 
     * 使用最小二乘求解端点切向长度，确保满足二次曲线兼容条件
     */
    inline glm::vec3 computeCSharp(glm::vec3 A, glm::vec3 B, glm::vec3 dA, glm::vec3 dB)
    {
        glm::vec3 e = B - A;
        
        // 确保方向一致
        if (glm::dot(dA, dB) < 0.0f)
            dB = -dB;
        
        // 构建 2x2 线性系统: G * [lA, lB]^T = r
        // 其中 G = [dA·dA, dA·dB; dA·dB, dB·dB]
        //      r = [2e·dA, 2e·dB]
        float G00 = glm::dot(dA, dA);
        float G01 = glm::dot(dA, dB);
        float G11 = glm::dot(dB, dB);
        float r0 = 2.0f * glm::dot(e, dA);
        float r1 = 2.0f * glm::dot(e, dB);
        
        // 正则化
        float lambda = 1e-6f;
        G00 += lambda;
        G11 += lambda;
        
        // 求解
        float det = G00 * G11 - G01 * G01;
        if (std::abs(det) < 1e-12f)
        {
            // 病态，退回边方向
            return glm::vec3(0.0f);
        }
        
        float lA = (G11 * r0 - G01 * r1) / det;
        float lB = (-G01 * r0 + G00 * r1) / det;
        
        glm::vec3 T_A = lA * dA;
        glm::vec3 T_B = lB * dB;
        
        // c_sharp = (T_B - T_A) / 2
        glm::vec3 c_sharp = 0.5f * (T_B - T_A);
        
        // 过冲钳制
        float eLen = glm::length(e);
        float cLen = glm::length(c_sharp);
        float maxC = 2.0f * eLen;
        if (cLen > maxC)
        {
            c_sharp = c_sharp * (maxC / cLen);
        }
        
        return c_sharp;
    }
    
    /**
     * @brief 为所有裂隙边计算 c_sharp
     */
    inline EnhancedNagataData computeCSharpForEdges(
        const std::map<EdgeKey, CreaseEdgeInfo>& creaseEdges)
    {
        EnhancedNagataData data;
        
        for (const auto& [edgeKey, info] : creaseEdges)
        {
            glm::vec3 e = info.B - info.A;
            
            // 计算端点折痕方向
            glm::vec3 dA = computeCreaseDirection(info.n_A_L, info.n_A_R, e);
            glm::vec3 dB = computeCreaseDirection(info.n_B_L, info.n_B_R, e);
            
            // 计算 c_sharp
            glm::vec3 c_sharp = computeCSharp(info.A, info.B, dA, dB);
            
            data.c_sharps[edgeKey] = c_sharp;
        }
        
        return data;
    }
    
    // ============================================================
    // 主入口 API
    // ============================================================
    
    /**
     * @brief 计算或加载增强数据
     * 
     * - 如果 .eng 缓存存在，直接加载
     * - 否则计算并保存缓存
     */
    inline EnhancedNagataData computeOrLoadEnhancedData(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        const std::string& nsmPath,
        bool saveCache = true)
    {
        std::string engPath = getEngFilepath(nsmPath);
        
        // 尝试加载缓存
        EnhancedNagataData data;
        if (loadEnhancedData(engPath, data))
        {
            return data;
        }
        
        // 计算
        std::cout << "NagataEnhanced: 检测裂隙边..." << std::endl;
        auto creaseEdges = detectCreaseEdges(vertices, faces, faceNormals);
        std::cout << "NagataEnhanced: 发现 " << creaseEdges.size() << " 条裂隙边" << std::endl;
        
        if (creaseEdges.empty())
        {
            return data;  // 空数据
        }
        
        std::cout << "NagataEnhanced: 计算共享边界系数..." << std::endl;
        data = computeCSharpForEdges(creaseEdges);
        
        // 保存缓存
        if (saveCache)
        {
            saveEnhancedData(engPath, data);
        }
        
        return data;
    }
    
    // ============================================================
    // 增强曲面求值
    // ============================================================
    
    /**
     * @brief Smoothstep 权重函数 (五次多项式)
     */
    inline float smoothstep(float t)
    {
        t = std::clamp(t, 0.0f, 1.0f);
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }
    
    /**
     * @brief 增强版 Nagata 曲面求值
     * 
     * @param patch 原始 Nagata 面片数据
     * @param u 参数 u
     * @param v 参数 v (满足 0 <= v <= u <= 1)
     * @param c_sharp_1 边1的 c_sharp (v0-v1)
     * @param c_sharp_2 边2的 c_sharp (v1-v2)
     * @param c_sharp_3 边3的 c_sharp (v0-v2)
     * @param isCrease 三条边是否为裂隙边
     * @param d0 影响宽度 (0.05 ~ 0.15)
     */
    inline glm::vec3 evaluateSurfaceEnhanced(
        const NagataPatch::NagataPatchData& patch,
        float u, float v,
        glm::vec3 c_sharp_1, glm::vec3 c_sharp_2, glm::vec3 c_sharp_3,
        std::array<bool, 3> isCrease,
        float d0 = 0.1f)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];
        
        // 原始系数
        glm::vec3 c1_orig = patch.curvatureCoeffs[0];
        glm::vec3 c2_orig = patch.curvatureCoeffs[1];
        glm::vec3 c3_orig = patch.curvatureCoeffs[2];
        
        // 计算到各边的距离参数
        float d1 = v;           // 边1 (v=0)
        float d2 = 1.0f - u;    // 边2 (u=1)
        float d3 = u - v;       // 边3 (u=v)
        
        // 计算有效系数
        auto blendCoeff = [&](glm::vec3 c_orig, glm::vec3 c_sharp, float d, bool isCreaseEdge) -> glm::vec3
        {
            if (!isCreaseEdge)
                return c_orig;
            
            float s = std::clamp(d / d0, 0.0f, 1.0f);
            float w = smoothstep(s);  // w=0 at edge, w=1 far from edge
            return (1.0f - w) * c_sharp + w * c_orig;
        };
        
        glm::vec3 c1_eff = blendCoeff(c1_orig, c_sharp_1, d1, isCrease[0]);
        glm::vec3 c2_eff = blendCoeff(c2_orig, c_sharp_2, d2, isCrease[1]);
        glm::vec3 c3_eff = blendCoeff(c3_orig, c_sharp_3, d3, isCrease[2]);
        
        // Nagata 曲面公式
        float oneMinusU = 1.0f - u;
        float uMinusV = u - v;
        
        return x00 * oneMinusU + x10 * uMinusV + x11 * v
             - c1_eff * oneMinusU * uMinusV
             - c2_eff * uMinusV * v
             - c3_eff * oneMinusU * v;
    }
}
}

#endif // NAGATA_ENHANCED_H
