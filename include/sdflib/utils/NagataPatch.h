#ifndef NAGATA_PATCH_H
#define NAGATA_PATCH_H

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <limits>

namespace sdflib
{
namespace NagataPatch
{
    // Forward declaration
    inline glm::vec3 computeCurvature(glm::vec3 d, glm::vec3 n0, glm::vec3 n1);
    /**
     * @brief Nagata曲面三角形数据结构
     * 
     * 存储三角形三个顶点和三条边的曲率系数
     * 用于Nagata二次插值曲面计算
     */
    struct NagataPatchData
    {
        NagataPatchData() {}
        
        /**
         * @brief 从三角形顶点和法向量构造Nagata曲面
         * 
         * @param v0 第一个顶点
         * @param v1 第二个顶点
         * @param v2 第三个顶点
         * @param n0 第一个顶点的法向量
         * @param n1 第二个顶点的法向量
         * @param n2 第三个顶点的法向量
         */
        NagataPatchData(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
                        glm::vec3 n0, glm::vec3 n1, glm::vec3 n2)
        {
            vertices[0] = v0;
            vertices[1] = v1;
            vertices[2] = v2;
            normals[0] = n0;
            normals[1] = n1;
            normals[2] = n2;
            
            // 计算三条边的曲率系数
            curvatureCoeffs[0] = computeCurvature(v1 - v0, n0, n1); // 边v0->v1
            curvatureCoeffs[1] = computeCurvature(v2 - v1, n1, n2); // 边v1->v2
            curvatureCoeffs[2] = computeCurvature(v2 - v0, n0, n2); // 边v0->v2
        }
        
        std::array<glm::vec3, 3> vertices;      // 三角形三个顶点
        std::array<glm::vec3, 3> normals;       // 三个顶点的法向量
        std::array<glm::vec3, 3> curvatureCoeffs; // 三条边的曲率系数 (c1, c2, c3)
    };
    
    /**
     * @brief 计算曲率系数 (基于Nagata 2005算法)
     * 
     * 根据方向向量和两个端点的法向量计算曲率系数
     * 当法向量接近平行时，退化为线性插值 (c = 0)
     * 
     * @param d 方向向量 (从点0到点1)
     * @param n0 点0的法向量
     * @param n1 点1的法向量
     * @return glm::vec3 曲率系数向量
     */
    inline glm::vec3 computeCurvature(glm::vec3 d, glm::vec3 n0, glm::vec3 n1)
    {
        // 角度容差：当法向量夹角小于0.1度时，认为是平行的
        // cos(0.1 deg) ≈ 0.9999984769
        static const float angleTol = 0.9999984769f;
        
        // 计算中间变量
        glm::vec3 v = 0.5f * (n0 + n1);
        glm::vec3 Deltav = 0.5f * (n0 - n1);
        
        float dv = glm::dot(d, v);
        float dDeltav = glm::dot(d, Deltav);
        
        float Deltac = glm::dot(n0, Deltav);
        float c = 1.0f - 2.0f * Deltac;
        
        // 检查法向量是否接近平行
        if (std::abs(c) <= angleTol)
        {
            // 正常情况：计算曲率系数
            return (dDeltav / (1.0f - Deltac)) * v + (dv / Deltac) * Deltav;
        }
        else
        {
            // 退化情况：法向量几乎平行，退化为线性插值
            return glm::vec3(0.0f);
        }
    }
    
    /**
     * @brief 在Nagata曲面上求值
     * 
     * Nagata曲面参数化公式：
     * x(u,v) = x00(1-u) + x10(u-v) + x11v 
     *        - c1(1-u)(u-v) - c2(u-v)v - c3(1-u)v
     * 
     * 其中参数域为三角形：0 <= v <= u <= 1
     * 
     * @param patch Nagata曲面数据
     * @param u 第一个参数 [0,1]
     * @param v 第二个参数 [0,u]
     * @return glm::vec3 曲面上的点坐标
     */
    inline glm::vec3 evaluateSurface(const NagataPatchData& patch, float u, float v)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];
        const glm::vec3& c1 = patch.curvatureCoeffs[0];
        const glm::vec3& c2 = patch.curvatureCoeffs[1];
        const glm::vec3& c3 = patch.curvatureCoeffs[2];
        
        float oneMinusU = 1.0f - u;
        float uMinusV = u - v;
        
        return x00 * oneMinusU + x10 * uMinusV + x11 * v
             - c1 * oneMinusU * uMinusV
             - c2 * uMinusV * v
             - c3 * oneMinusU * v;
    }
    
    /**
     * @brief 计算Nagata曲面在参数(u,v)处的偏导数
     * 
     * @param patch Nagata曲面数据
     * @param u 第一个参数
     * @param v 第二个参数
     * @param dXdu 输出：对u的偏导数
     * @param dXdv 输出：对v的偏导数
     */
    inline void evaluateDerivatives(const NagataPatchData& patch, float u, float v,
                                    glm::vec3& dXdu, glm::vec3& dXdv)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];
        const glm::vec3& c1 = patch.curvatureCoeffs[0];
        const glm::vec3& c2 = patch.curvatureCoeffs[1];
        const glm::vec3& c3 = patch.curvatureCoeffs[2];
        
        // dX/du = -x00 + x10 + c1(1-u) - c1(u-v) + c2v + c3v
        //       = -x00 + x10 + c1(1-2u+v) + c2v + c3v
        dXdu = -x00 + x10 + c1 * (1.0f - 2.0f * u + v) + c2 * v + c3 * v;
        
        // dX/dv = -x10 + x11 + c1(1-u) - c2(u-2v) - c3(1-u)
        //       = -x10 + x11 + (c1-c3)(1-u) - c2(u-2v)
        dXdv = -x10 + x11 + (c1 - c3) * (1.0f - u) - c2 * (u - 2.0f * v);
    }
    
    /**
     * @brief 计算Nagata曲面在参数(u,v)处的法向量
     * 
     * @param patch Nagata曲面数据
     * @param u 第一个参数
     * @param v 第二个参数
     * @return glm::vec3 单位法向量
     */
    inline glm::vec3 evaluateNormal(const NagataPatchData& patch, float u, float v)
    {
        glm::vec3 dXdu, dXdv;
        evaluateDerivatives(patch, u, v, dXdu, dXdv);
        
        glm::vec3 normal = glm::cross(dXdu, dXdv);
        float len = glm::length(normal);
        
        if (len > 1e-10f)
        {
            return normal / len;
        }
        else
        {
            // 退化情况：返回顶点法向量的插值
            float w0 = 1.0f - u;
            float w1 = u - v;
            float w2 = v;
            glm::vec3 interpNormal = w0 * patch.normals[0] + w1 * patch.normals[1] + w2 * patch.normals[2];
            return glm::normalize(interpNormal);
        }
    }
    
    /**
     * @brief 使用牛顿迭代法计算点到Nagata曲面的最近点
     * 
     * @param point 查询点
     * @param patch Nagata曲面数据
     * @param nearestPoint 输出：曲面上的最近点
     * @param minU 输出：最近点的u参数
     * @param minV 输出：最近点的v参数
     * @param maxIterations 最大迭代次数
     * @return float 距离的平方
     */
    inline float findNearestPointOnNagataPatch(glm::vec3 point, const NagataPatchData& patch,
                                                glm::vec3& nearestPoint, float& minU, float& minV,
                                                int maxIterations = 10)
    {
        // 初始猜测：使用三角形重心坐标
        minU = 0.333f;
        minV = 0.166f;
        
        float minDistSq = std::numeric_limits<float>::max();
        
        // 多初始点策略：在三角形内均匀采样几个初始点
        const std::array<std::pair<float, float>, 7> initialSamples = {{
            {0.5f, 0.25f},    // 重心
            {0.0f, 0.0f},     // 顶点0
            {1.0f, 0.0f},     // 顶点1
            {1.0f, 1.0f},     // 顶点2
            {0.5f, 0.0f},     // 边0-1中点
            {1.0f, 0.5f},     // 边1-2中点
            {0.5f, 0.5f}      // 边0-2中点
        }};
        
        for (const auto& sample : initialSamples)
        {
            float u = sample.first;
            float v = sample.second;
            
            // 确保参数在有效域内
            if (v < 0.0f) v = 0.0f;
            if (v > u) v = u;
            if (u < 0.0f) u = 0.0f;
            if (u > 1.0f) u = 1.0f;
            
            // 牛顿迭代
            for (int iter = 0; iter < maxIterations; ++iter)
            {
                glm::vec3 surfacePoint = evaluateSurface(patch, u, v);
                glm::vec3 diff = point - surfacePoint;
                
                glm::vec3 dXdu, dXdv;
                evaluateDerivatives(patch, u, v, dXdu, dXdv);
                
                // 计算梯度 (目标函数是距离平方的一半)
                // f(u,v) = 0.5 * |point - surface(u,v)|^2
                // grad_f = -[dXdu·diff, dXdv·diff]
                float gradU = -glm::dot(dXdu, diff);
                float gradV = -glm::dot(dXdv, diff);
                
                // 计算Hessian矩阵
                // H = [dXdu·dXdu - d2Xdu2·diff, dXdu·dXdv - d2Xdudv·diff]
                //     [dXdu·dXdv - d2Xdudv·diff, dXdv·dXdv - d2Xdv2·diff]
                // 简化：忽略二阶导数项
                float H11 = glm::dot(dXdu, dXdu);
                float H12 = glm::dot(dXdu, dXdv);
                float H22 = glm::dot(dXdv, dXdv);
                
                // 求解H * delta = -grad
                float det = H11 * H22 - H12 * H12;
                if (std::abs(det) < 1e-10f) break;
                
                float deltaU = (H22 * gradU - H12 * gradV) / det;
                float deltaV = (-H12 * gradU + H11 * gradV) / det;
                
                // 阻尼牛顿法
                float stepSize = 1.0f;
                float newU = u + stepSize * deltaU;
                float newV = v + stepSize * deltaV;
                
                // 投影到有效域
                if (newV < 0.0f) newV = 0.0f;
                if (newV > newU) newV = newU;
                if (newU < 0.0f) newU = 0.0f;
                if (newU > 1.0f) newU = 1.0f;
                
                // 检查收敛
                if (std::abs(newU - u) < 1e-6f && std::abs(newV - v) < 1e-6f)
                {
                    u = newU;
                    v = newV;
                    break;
                }
                
                u = newU;
                v = newV;
            }
            
            // 计算当前解的距离
            glm::vec3 surfacePoint = evaluateSurface(patch, u, v);
            float distSq = glm::dot(point - surfacePoint, point - surfacePoint);
            
            if (distSq < minDistSq)
            {
                minDistSq = distSq;
                minU = u;
                minV = v;
                nearestPoint = surfacePoint;
            }
        }
        
        // 最终计算
        nearestPoint = evaluateSurface(patch, minU, minV);
        minDistSq = glm::dot(point - nearestPoint, point - nearestPoint);
        
        return minDistSq;
    }
    
    /**
     * @brief 计算点到Nagata曲面的有符号距离
     * 
     * @param point 查询点
     * @param patch Nagata曲面数据
     * @param outNearestPoint 输出：最近点（可选）
     * @return float 有符号距离
     */
    inline float getSignedDistPointAndNagataPatch(glm::vec3 point, const NagataPatchData& patch,
                                                   glm::vec3* outNearestPoint = nullptr)
    {
        glm::vec3 nearestPoint;
        float u, v;
        
        float distSq = findNearestPointOnNagataPatch(point, patch, nearestPoint, u, v);
        
        // 计算法向量确定符号
        glm::vec3 normal = evaluateNormal(patch, u, v);
        glm::vec3 diff = point - nearestPoint;
        
        float sign = glm::dot(diff, normal) >= 0.0f ? 1.0f : -1.0f;
        
        if (outNearestPoint != nullptr)
        {
            *outNearestPoint = nearestPoint;
        }
        
        return sign * std::sqrt(distSq);
    }
    
    /**
     * @brief 计算点到Nagata曲面的无符号距离
     * 
     * @param point 查询点
     * @param patch Nagata曲面数据
     * @param outNearestPoint 输出：最近点（可选）
     * @return float 无符号距离
     */
    inline float getUnsignedDistPointAndNagataPatch(glm::vec3 point, const NagataPatchData& patch,
                                                     glm::vec3* outNearestPoint = nullptr)
    {
        glm::vec3 nearestPoint;
        float u, v;
        
        float distSq = findNearestPointOnNagataPatch(point, patch, nearestPoint, u, v);
        
        if (outNearestPoint != nullptr)
        {
            *outNearestPoint = nearestPoint;
        }
        
        return std::sqrt(distSq);
    }
}
}

#endif // NAGATA_PATCH_H
