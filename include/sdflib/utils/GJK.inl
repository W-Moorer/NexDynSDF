/**
 * @file GJK.inl
 * @brief GJK算法的内联实现
 */

#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

namespace sdflib
{
namespace GJK
{

// 8个子节点偏移
inline const std::array<glm::vec3, 8> childrenOffsets = 
{
    glm::vec3(-1.0f, -1.0f, -1.0f),
    glm::vec3(1.0f, -1.0f, -1.0f),
    glm::vec3(-1.0f, 1.0f, -1.0f),
    glm::vec3(1.0f, 1.0f, -1.0f),
    glm::vec3(-1.0f, -1.0f, 1.0f),
    glm::vec3(1.0f, -1.0f, 1.0f),
    glm::vec3(-1.0f, 1.0f, 1.0f),
    glm::vec3(1.0f, 1.0f, 1.0f)
};

// 符号函数
inline glm::vec3 fsign(glm::vec3 a)
{
    return glm::vec3(
        (a.x > 0.0f) ? 1.0f : -1.0f,
        (a.y > 0.0f) ? 1.0f : -1.0f,
        (a.z > 0.0f) ? 1.0f : -1.0f
    );
}

// 线段的最近点方向
template<typename S>
inline bool getLineOriginDirection(S& simplex, glm::vec3& outDirection)
{
    const glm::vec3 ab = simplex[1] - simplex[0];
    const glm::vec3 ao = -simplex[0];

    if(glm::dot(ab, ao) > 0.0f)
    {
        outDirection = glm::cross(ab, glm::cross(ao, ab));
    }
    else
    {
        simplex.type = SimplexType::POINT;
        outDirection = ao;
    }
    return false;
}

// 三角形的最近点方向
template<typename S>
inline bool getTriangleOriginDirection(S& simplex, glm::vec3& outDirection)
{
    glm::vec3 ab = simplex[1] - simplex[0];
    glm::vec3 ac = simplex[2] - simplex[0];
    glm::vec3 ao = -simplex[0];
    glm::vec3 n = glm::cross(ab, ac);

    if(glm::dot(glm::cross(n, ac), ao) > 0.0f)
    {
        if(glm::dot(ac, ao) > 0.0f)
        {
            simplex.type = SimplexType::LINE;
            simplex.points[1] = simplex.points[2];
            outDirection = glm::cross(ac, glm::cross(ao, ac));
        }
        else
        {
            simplex.type = SimplexType::LINE;
            return getLineOriginDirection(simplex, outDirection);
        }
    }
    else if(glm::dot(glm::cross(ab, n), ao) > 0.0f)
    {
        simplex.type = SimplexType::LINE;
        return getLineOriginDirection(simplex, outDirection);
    }
    else
    {
        if(glm::dot(n, ao) > 0.0f)
        {
            outDirection = n;
        }
        else
        {
            auto aux = simplex.points[1];
            simplex.points[1] = simplex.points[2];
            simplex.points[2] = aux;
            outDirection = -n;
        }
    }
    return false;
}

// 四面体的最近点方向（情况2）
template<typename S>
inline bool getTetrahedronOriginDirectionCase2(S& simplex, glm::vec3& outDirection)
{
    glm::vec3 ab = simplex[1] - simplex[0];
    glm::vec3 ac = simplex[2] - simplex[0];
    glm::vec3 ad = simplex[3] - simplex[0];
    glm::vec3 ao = -simplex[0];

    glm::vec3 abc = glm::cross(ab, ac);
    glm::vec3 acd = glm::cross(ac, ad);
    glm::vec3 adb = glm::cross(ad, ab);

    bool enter = false;
    bool acOutAbc = glm::dot(glm::cross(abc, ac), ao) > 0.0f;
    bool acOutAcd = glm::dot(glm::cross(ac, acd), ao) > 0.0f;

    if(acOutAbc && acOutAcd)
    {
        if(glm::dot(ac, ao) > 0.0f)
        {
            simplex.type = SimplexType::LINE;
            simplex.points[1] = simplex.points[2];
            outDirection = glm::cross(ac, glm::cross(ao, ac));
            return false;
        }
        enter = true;
    }

    bool abOutAbc = glm::dot(glm::cross(ab, abc), ao) > 0.0f;
    bool abOutAdb = glm::dot(glm::cross(adb, ab), ao) > 0.0f;

    if(abOutAbc && abOutAdb)
    {
        if(glm::dot(ab, ao) > 0.0f)
        {
            simplex.type = SimplexType::LINE;
            outDirection = glm::cross(ab, glm::cross(ao, ab));
            return false;
        }
        enter = true;
    }

    if(glm::dot(abc, ao) > 0.0f && !acOutAbc && !abOutAbc)
    {
        simplex.type = SimplexType::TRIANGLE;
        outDirection = abc;
        return false;
    }

    bool adOutAcd = glm::dot(glm::cross(acd, ad), ao) > 0.0f;
    bool adOutAdb = glm::dot(glm::cross(ad, adb), ao) > 0.0f;
    if(adOutAcd && adOutAdb)
    {
        if(glm::dot(ad, ao) > 0.0f)
        {
            simplex.type = SimplexType::LINE;
            simplex.points[1] = simplex.points[3];
            outDirection = glm::cross(ad, glm::cross(ao, ad));
            return false;
        }
        enter = true;
    }

    if(glm::dot(acd, ao) > 0.0f && !acOutAcd && !adOutAcd)
    {
        simplex.type = SimplexType::TRIANGLE;
        simplex.points[1] = simplex.points[2];
        simplex.points[2] = simplex.points[3];
        outDirection = acd;
        return false;
    }

    if(glm::dot(adb, ao) > 0.0f && !adOutAdb && !abOutAdb)
    {
        simplex.type = SimplexType::TRIANGLE;
        simplex.points[2] = simplex.points[1];
        simplex.points[1] = simplex.points[3];
        outDirection = adb;
        return false;
    }

    if(enter)
    {
        simplex.type = SimplexType::POINT;
        outDirection = ao;
        return false;
    }

    return true;
}

// 获取原点到单纯形的方向
template<typename S>
inline bool getOriginDirection(S& simplex, glm::vec3& outDirection, float dotLastEnterPoint)
{
    switch(simplex.type)
    {
        case SimplexType::LINE:
            return getLineOriginDirection(simplex, outDirection);
        case SimplexType::TRIANGLE:
            return getTriangleOriginDirection(simplex, outDirection);
        case SimplexType::TETRAHEDRON:
            return getTetrahedronOriginDirectionCase2(simplex, outDirection);
    }
    return false;
}

// 在凸集中找到支持点
inline glm::vec3 findFurthestPoint(const std::vector<glm::vec3>& e, glm::vec3 direction)
{
    float maxValue = glm::dot(e[0], direction);
    int maxIndex = 0;
    for(size_t i = 1; i < e.size(); i++)
    {
        const float value = glm::dot(e[i], direction);
        if(value > maxValue)
        {
            maxValue = value;
            maxIndex = static_cast<int>(i);
        }
    }
    return e[maxIndex];
}

// Minkowski差的支持点
inline glm::vec3 findFurthestPoint(const std::vector<glm::vec3>& e1, const std::vector<glm::vec3>& e2, glm::vec3 direction)
{
    return findFurthestPoint(e1, direction) - findFurthestPoint(e2, -direction);
}

// 计算AABB与三角形之间的支持点
inline glm::vec3 findFurthestPoint(const glm::vec3& quadSize, const std::array<glm::vec3, 3>& triangle, const glm::vec3& direction)
{
    const float d1 = glm::dot(triangle[0], direction);
    const float d2 = glm::dot(triangle[1], direction);
    const float d3 = glm::dot(triangle[2], direction);
    const glm::vec3 quadPos = fsign(-direction) * quadSize;

    if(d1 > d2)
    {
        return (d1 > d3) ? triangle[0] - quadPos : triangle[2] - quadPos;
    }
    else
    {
        return (d2 > d3) ? triangle[1] - quadPos : triangle[2] - quadPos;
    }
}

// 带索引追踪的支持点计算
inline TrackedSimplex::Point findFurthestPointAndIndices(const glm::vec3& quadSize, const std::array<glm::vec3, 3>& triangle, const glm::vec3& direction)
{
    const float d1 = glm::dot(triangle[0], direction);
    const float d2 = glm::dot(triangle[1], direction);
    const float d3 = glm::dot(triangle[2], direction);

    const glm::vec3 quadPos = fsign(-direction) * quadSize;
    const uint32_t quadIndex = ((direction.z < 0) ? 4 : 0) + 
                               ((direction.y < 0) ? 2 : 0) + 
                               ((direction.x < 0) ? 1 : 0);

    if(d1 > d2)
    {
        return (d1 > d3) 
                ? TrackedSimplex::Point{triangle[0] - quadPos, std::make_pair(0, quadIndex)}
                : TrackedSimplex::Point{triangle[2] - quadPos, std::make_pair(2, quadIndex)};
    }
    else
    {
        return (d2 > d3) 
                ? TrackedSimplex::Point{triangle[1] - quadPos, std::make_pair(1, quadIndex)} 
                : TrackedSimplex::Point{triangle[2] - quadPos, std::make_pair(2, quadIndex)};
    }
}

// 计算点到AABB的最大距离平方
inline float sqMaxDistToQuad(const glm::vec3& point, const glm::vec3& quadSize)
{    
    glm::vec3 aux = point - glm::sign(-point) * quadSize;
    return glm::dot(aux, aux);
}

// 在球体集合中找到支持点
inline glm::vec3 findFurthestPoint(const std::vector<std::pair<glm::vec3, float>>& spheresShape, const glm::vec3& direction)
{
    float maxValue = glm::dot(spheresShape[0].first, direction) + spheresShape[0].second;
    int maxIndex = 0;
    for(size_t i = 1; i < spheresShape.size(); i++)
    {
        const float value = glm::dot(spheresShape[i].first, direction) + spheresShape[i].second;
        if(value > maxValue)
        {
            maxValue = value;
            maxIndex = static_cast<int>(i);
        }
    }
    return spheresShape[maxIndex].first + spheresShape[maxIndex].second * direction;
}

// 在三角形中找到支持点
inline glm::vec3 findFurthestPoint(const std::array<glm::vec3, 3>& triangle, const glm::vec3& direction)
{
    const float d1 = glm::dot(triangle[0], direction);
    const float d2 = glm::dot(triangle[1], direction);
    const float d3 = glm::dot(triangle[2], direction);

    if(d1 > d2) return (d1 > d3) ? triangle[0] : triangle[2];
    else return (d2 > d3) ? triangle[1] : triangle[2];
}

// 球体集合与三角形的支持点（Minkowski差）
inline glm::vec3 findFurthestPoint(const std::vector<std::pair<glm::vec3, float>>& spheresShape, 
                                   const std::array<glm::vec3, 3>& triangle,
                                   const glm::vec3& direction)
{
    return findFurthestPoint(spheresShape, direction) - findFurthestPoint(triangle, -direction);
}

// 八叉树节点的支持点（8个角点半径定义）
inline glm::vec3 findFurthestPoint(float halfNodeSize, const std::array<float, 8>& vertRadius, const glm::vec3& direction)
{
    float maxValue = glm::dot(glm::vec3(-halfNodeSize), direction) + vertRadius[0];
    int maxIndex = 0;
    for(int i = 1; i < 8; i++)
    {
        const float value = glm::dot(childrenOffsets[i] * halfNodeSize, direction) + vertRadius[i];
        if(value > maxValue)
        {
            maxValue = value;
            maxIndex = i;
        }
    }
    return childrenOffsets[maxIndex] * halfNodeSize + vertRadius[maxIndex] * direction;
}

// 八叉树节点与三角形的支持点
inline glm::vec3 findFurthestPoint(float halfNodeSize,
                                   const std::array<float, 8>& vertRadius, 
                                   const std::array<glm::vec3, 3>& triangle,
                                   const glm::vec3& direction)
{
    return findFurthestPoint(halfNodeSize, vertRadius, direction) - findFurthestPoint(triangle, -direction);
}

} // namespace GJK
} // namespace sdflib
