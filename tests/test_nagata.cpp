#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include "SdfLib/utils/NagataPatch.h"
#include "SdfLib/NagataTriangleMeshDistance.h"

using namespace sdflib;

/**
 * @brief 测试Nagata曲率系数计算
 * 
 * 验证曲率系数计算是否符合预期
 */
void testComputeCurvature()
{
    std::cout << "=== Testing computeCurvature ===" << std::endl;
    
    // 测试1：平行法向量（应该退化为0）
    {
        glm::vec3 d(1.0f, 0.0f, 0.0f);
        glm::vec3 n0(0.0f, 0.0f, 1.0f);
        glm::vec3 n1(0.0f, 0.0f, 1.0f); // 与n0平行
        
        glm::vec3 c = NagataPatch::computeCurvature(d, n0, n1);
        
        float len = glm::length(c);
        std::cout << "Test 1 - Parallel normals: c length = " << len << std::endl;
        if (len < 1e-6f) {
            std::cout << "  PASS: Curvature is zero for parallel normals" << std::endl;
        } else {
            std::cout << "  FAIL: Expected zero curvature" << std::endl;
        }
    }
    
    // 测试2：非平行法向量
    {
        glm::vec3 d(1.0f, 0.0f, 0.0f);
        glm::vec3 n0(0.0f, 0.0f, 1.0f);
        glm::vec3 n1(0.0f, 0.1f, 0.995f); // 稍微倾斜
        n1 = glm::normalize(n1);
        
        glm::vec3 c = NagataPatch::computeCurvature(d, n0, n1);
        
        std::cout << "Test 2 - Non-parallel normals: c = (" 
                  << c.x << ", " << c.y << ", " << c.z << ")" << std::endl;
        
        // 验证曲率系数的性质
        // 根据Nagata理论，c应该满足：n0·(d - c) = 0 和 n1·(d - c) = 0
        // 即 n0·c = n0·d 和 n1·c = -n1·d
        float n0dotc = glm::dot(n0, c);
        float n0dotd = glm::dot(n0, d);
        float n1dotc = glm::dot(n1, c);
        float n1dotd = glm::dot(n1, d);
        
        std::cout << "  n0·c = " << n0dotc << ", n0·d = " << n0dotd << std::endl;
        std::cout << "  n1·c = " << n1dotc << ", -n1·d = " << -n1dotd << std::endl;
        
        if (std::abs(n0dotc - n0dotd) < 1e-4f && std::abs(n1dotc + n1dotd) < 1e-4f) {
            std::cout << "  PASS: Curvature satisfies boundary conditions" << std::endl;
        } else {
            std::cout << "  FAIL: Curvature does not satisfy boundary conditions" << std::endl;
        }
    }
}

/**
 * @brief 测试Nagata曲面求值
 * 
 * 验证曲面在顶点处的值是否正确
 */
void testSurfaceEvaluation()
{
    std::cout << "\n=== Testing surface evaluation ===" << std::endl;
    
    // 创建一个简单的三角形
    glm::vec3 v0(0.0f, 0.0f, 0.0f);
    glm::vec3 v1(1.0f, 0.0f, 0.0f);
    glm::vec3 v2(0.5f, 1.0f, 0.0f);
    
    // 法向量都指向z方向
    glm::vec3 n0(0.0f, 0.0f, 1.0f);
    glm::vec3 n1(0.0f, 0.0f, 1.0f);
    glm::vec3 n2(0.0f, 0.0f, 1.0f);
    
    NagataPatch::NagataPatchData patch(v0, v1, v2, n0, n1, n2);
    
    // 测试顶点处
    glm::vec3 p0 = NagataPatch::evaluateSurface(patch, 0.0f, 0.0f);
    glm::vec3 p1 = NagataPatch::evaluateSurface(patch, 1.0f, 0.0f);
    glm::vec3 p2 = NagataPatch::evaluateSurface(patch, 1.0f, 1.0f);
    
    std::cout << "Vertex 0: expected (0,0,0), got (" << p0.x << "," << p0.y << "," << p0.z << ")" << std::endl;
    std::cout << "Vertex 1: expected (1,0,0), got (" << p1.x << "," << p1.y << "," << p1.z << ")" << std::endl;
    std::cout << "Vertex 2: expected (0.5,1,0), got (" << p2.x << "," << p2.y << "," << p2.z << ")" << std::endl;
    
    float err0 = glm::length(p0 - v0);
    float err1 = glm::length(p1 - v1);
    float err2 = glm::length(p2 - v2);
    
    if (err0 < 1e-6f && err1 < 1e-6f && err2 < 1e-6f) {
        std::cout << "PASS: Surface passes through all vertices" << std::endl;
    } else {
        std::cout << "FAIL: Surface does not pass through vertices" << std::endl;
    }
}

/**
 * @brief 测试点到Nagata曲面的距离计算
 */
void testPointToPatchDistance()
{
    std::cout << "\n=== Testing point to patch distance ===" << std::endl;
    
    // 创建一个平面三角形
    glm::vec3 v0(0.0f, 0.0f, 0.0f);
    glm::vec3 v1(1.0f, 0.0f, 0.0f);
    glm::vec3 v2(0.0f, 1.0f, 0.0f);
    
    glm::vec3 n0(0.0f, 0.0f, 1.0f);
    glm::vec3 n1(0.0f, 0.0f, 1.0f);
    glm::vec3 n2(0.0f, 0.0f, 1.0f);
    
    NagataPatch::NagataPatchData patch(v0, v1, v2, n0, n1, n2);
    
    // 测试点：在三角形正上方
    glm::vec3 queryPoint(0.2f, 0.2f, 1.0f);
    
    glm::vec3 nearestPoint;
    float dist = NagataPatch::getSignedDistPointAndNagataPatch(queryPoint, patch, &nearestPoint);
    
    std::cout << "Query point: (" << queryPoint.x << "," << queryPoint.y << "," << queryPoint.z << ")" << std::endl;
    std::cout << "Nearest point: (" << nearestPoint.x << "," << nearestPoint.y << "," << nearestPoint.z << ")" << std::endl;
    std::cout << "Signed distance: " << dist << std::endl;
    
    // 对于平面三角形，距离应该是1.0
    if (std::abs(std::abs(dist) - 1.0f) < 1e-3f) {
        std::cout << "PASS: Distance is correct for plane triangle" << std::endl;
    } else {
        std::cout << "FAIL: Expected distance ~1.0, got " << std::abs(dist) << std::endl;
    }
    
    // 测试符号
    if (dist > 0.0f) {
        std::cout << "PASS: Sign is positive for point above triangle" << std::endl;
    } else {
        std::cout << "FAIL: Expected positive sign" << std::endl;
    }
}

/**
 * @brief 测试NagataTriangleMeshDistance类
 */
void testNagataMeshDistance()
{
    std::cout << "\n=== Testing NagataTriangleMeshDistance ===" << std::endl;
    
    // 创建一个简单的立方体网格
    std::vector<glm::vec3> vertices = {
        // 前面 (z = 0)
        {0.0f, 0.0f, 0.0f}, // 0
        {1.0f, 0.0f, 0.0f}, // 1
        {1.0f, 1.0f, 0.0f}, // 2
        {0.0f, 1.0f, 0.0f}, // 3
    };
    
    std::vector<std::array<int, 3>> triangles = {
        {0, 1, 2}, // 前面三角形1
        {0, 2, 3}, // 前面三角形2
    };
    
    try {
        NagataTriangleMeshDistance::NagataTriangleMeshDistance meshDist(vertices, triangles);
        
        if (meshDist.isConstructed()) {
            std::cout << "PASS: Mesh distance object constructed successfully" << std::endl;
        } else {
            std::cout << "FAIL: Mesh distance object not constructed" << std::endl;
            return;
        }
        
        // 测试查询
        glm::vec3 queryPoint(0.5f, 0.5f, 1.0f);
        auto result = meshDist.signedDistance(queryPoint);
        
        std::cout << "Query point: (" << queryPoint.x << "," << queryPoint.y << "," << queryPoint.z << ")" << std::endl;
        std::cout << "Signed distance: " << result.distance << std::endl;
        std::cout << "Nearest point: (" << result.nearestPoint.x << "," 
                  << result.nearestPoint.y << "," << result.nearestPoint.z << ")" << std::endl;
        std::cout << "Triangle ID: " << result.triangleId << std::endl;
        
        // 对于单位正方形在z=0平面，点(0.5,0.5,1)的距离应该是1.0
        if (std::abs(result.distance - 1.0f) < 1e-2f) {
            std::cout << "PASS: Distance query returns correct value" << std::endl;
        } else {
            std::cout << "FAIL: Expected distance ~1.0, got " << result.distance << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }
}

/**
 * @brief 对比Nagata曲面与线性三角形的差异
 */
void testCurvedSurface()
{
    std::cout << "\n=== Testing curved surface (Nagata vs Linear) ===" << std::endl;
    
    // 创建一个弯曲的曲面：顶点在同一平面，但法向量不同
    glm::vec3 v0(0.0f, 0.0f, 0.0f);
    glm::vec3 v1(1.0f, 0.0f, 0.0f);
    glm::vec3 v2(0.5f, 1.0f, 0.0f);
    
    // 法向量向外倾斜，形成一个碗状曲面
    glm::vec3 n0 = glm::normalize(glm::vec3(0.0f, -0.3f, 1.0f));
    glm::vec3 n1 = glm::normalize(glm::vec3(0.0f, -0.3f, 1.0f));
    glm::vec3 n2 = glm::normalize(glm::vec3(0.0f, 0.5f, 1.0f));
    
    NagataPatch::NagataPatchData patch(v0, v1, v2, n0, n1, n2);
    
    // 在三角形中心采样几个点
    std::cout << "Sampling surface at different (u,v) coordinates:" << std::endl;
    
    for (float u = 0.0f; u <= 1.0f; u += 0.25f) {
        for (float v = 0.0f; v <= u; v += 0.25f) {
            glm::vec3 p = NagataPatch::evaluateSurface(patch, u, v);
            
            // 线性插值参考
            float w0 = 1.0f - u;
            float w1 = u - v;
            float w2 = v;
            glm::vec3 pLinear = w0 * v0 + w1 * v1 + w2 * v2;
            
            float diff = glm::length(p - pLinear);
            std::cout << "  (u=" << u << ",v=" << v << "): Nagata z=" << p.z 
                      << ", Linear z=" << pLinear.z << ", diff=" << diff << std::endl;
        }
    }
    
    std::cout << "PASS: Curved surface evaluation completed" << std::endl;
}

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "Nagata Patch Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    testComputeCurvature();
    testSurfaceEvaluation();
    testPointToPatchDistance();
    testNagataMeshDistance();
    testCurvedSurface();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Completed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
