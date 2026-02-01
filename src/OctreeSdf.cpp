/**
 * @file OctreeSdf.cpp
 * @brief Adaptive octree-based SDF implementation
 */

#include "sdflib/OctreeSdf.h"
#include "sdflib/TriangleMeshDistance.h"
#include "sdflib/utils/Timer.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <stack>
#include <cmath>
#include <optional>

namespace sdflib
{

// Simple distance query using TriangleMeshDistance
class MeshDistanceQuery
{
public:
    MeshDistanceQuery(const Mesh& mesh)
    {
        std::vector<tmd::Vec3d> vertices;
        vertices.reserve(mesh.getVertices().size());
        for (const auto& v : mesh.getVertices())
        {
            vertices.push_back(tmd::Vec3d(v.x, v.y, v.z));
        }

        std::vector<std::array<int, 3>> triangles;
        triangles.reserve(mesh.getIndices().size() / 3);
        for (size_t i = 0; i < mesh.getIndices().size(); i += 3)
        {
            triangles.push_back({
                static_cast<int>(mesh.getIndices()[i]),
                static_cast<int>(mesh.getIndices()[i + 1]),
                static_cast<int>(mesh.getIndices()[i + 2])
            });
        }

        mDistanceQuery.construct(vertices, triangles);
    }

    float queryDistance(const glm::vec3& point, glm::vec3& gradient) const
    {
        tmd::Result result = mDistanceQuery.signed_distance({point.x, point.y, point.z});
        gradient = glm::vec3(
            result.nearest_point.v[0] - point.x,
            result.nearest_point.v[1] - point.y,
            result.nearest_point.v[2] - point.z
        );
        float len = glm::length(gradient);
        if (len > 1e-6f)
        {
            gradient /= len;
        }
        return static_cast<float>(result.distance);
    }

private:
    tmd::TriangleMeshDistance mDistanceQuery;
};

OctreeSdf::OctreeSdf(const Mesh& mesh, BoundingBox box,
                     uint32_t depth, uint32_t startDepth,
                     float terminationThreshold,
                     InitAlgorithm initAlgorithm,
                     uint32_t numThreads)
{
    buildOctree(mesh, box, depth, startDepth,
                TerminationRule::TRAPEZOIDAL_RULE,
                TerminationRuleParams::setTrapezoidalRuleParams(terminationThreshold),
                initAlgorithm, numThreads);
}

OctreeSdf::OctreeSdf(const Mesh& mesh, BoundingBox box,
                     uint32_t depth, uint32_t startDepth,
                     TerminationRule terminationRule,
                     TerminationRuleParams params,
                     InitAlgorithm initAlgorithm,
                     uint32_t numThreads)
{
    buildOctree(mesh, box, depth, startDepth, terminationRule, params,
                initAlgorithm, numThreads);
}

void OctreeSdf::buildOctree(const Mesh& mesh, BoundingBox box,
                            uint32_t depth, uint32_t startDepth,
                            TerminationRule terminationRule,
                            TerminationRuleParams params,
                            InitAlgorithm initAlgorithm,
                            uint32_t numThreads)
{
    mMaxDepth = depth;

    const glm::vec3 bbSize = box.getSize();
    const float maxSize = std::max(std::max(bbSize.x, bbSize.y), bbSize.z);
    mBox.min = box.getCenter() - 0.5f * maxSize;
    mBox.max = box.getCenter() + 0.5f * maxSize;

    mStartGridSize = 1 << startDepth;
    mStartGridXY = mStartGridSize * mStartGridSize;
    mStartGridCellSize = maxSize / static_cast<float>(mStartGridSize);

    SPDLOG_INFO("Building octree with depth {}, start depth {}", depth, startDepth);
    SPDLOG_INFO("Grid size: {}x{}x{}", mStartGridSize, mStartGridSize, mStartGridSize);

    // Create distance query
    MeshDistanceQuery distanceQuery(mesh);

    // Initialize uniform grid at start depth
    initUniformOctree(mesh, startDepth, depth);

    computeMinBorderValue();
    
    SPDLOG_INFO("Octree build complete. Nodes: {}", mOctreeData.size());
}

void OctreeSdf::initUniformOctree(const Mesh& mesh, uint32_t startDepth, uint32_t depth)
{
    MeshDistanceQuery distanceQuery(mesh);

    // Initialize start grid
    const uint32_t numStartCells = mStartGridSize * mStartGridSize * mStartGridSize;
    mOctreeData.resize(numStartCells);

    // For each cell in start grid
    for (uint32_t z = 0; z < mStartGridSize; z++)
    {
        for (uint32_t y = 0; y < mStartGridSize; y++)
        {
            for (uint32_t x = 0; x < mStartGridSize; x++)
            {
                const uint32_t idx = z * mStartGridXY + y * mStartGridSize + x;
                
                // Calculate cell center
                const glm::vec3 cellCenter = mBox.min + glm::vec3(
                    (static_cast<float>(x) + 0.5f) * mStartGridCellSize,
                    (static_cast<float>(y) + 0.5f) * mStartGridCellSize,
                    (static_cast<float>(z) + 0.5f) * mStartGridCellSize
                );

                // Sample distance at cell center
                glm::vec3 gradient;
                float distance = distanceQuery.queryDistance(cellCenter, gradient);

                // Store as leaf node with single value
                mOctreeData[idx].setLeaf();
                // Store distance in childrenIndex (lower 31 bits for leaf data index)
                union { float f; uint32_t i; } converter;
                converter.f = distance;
                mOctreeData[idx].setChildrenIndex(converter.i & 0x7FFFFFFF);
            }
        }
    }
}

float OctreeSdf::getDistance(glm::vec3 sample) const
{
    glm::vec3 fracPart = (sample - mBox.min) / mStartGridCellSize;
    glm::ivec3 startArrayPos = glm::floor(fracPart);
    fracPart = glm::fract(fracPart);

    if (startArrayPos.x < 0 || startArrayPos.x >= static_cast<int>(mStartGridSize) ||
        startArrayPos.y < 0 || startArrayPos.y >= static_cast<int>(mStartGridSize) ||
        startArrayPos.z < 0 || startArrayPos.z >= static_cast<int>(mStartGridSize))
    {
        return mBox.getDistance(sample) + mMinBorderValue;
    }

    const uint32_t idx = startArrayPos.z * mStartGridXY + 
                         startArrayPos.y * mStartGridSize + 
                         startArrayPos.x;

    if (idx >= mOctreeData.size())
    {
        return mMinBorderValue;
    }

    const OctreeNode* node = &mOctreeData[idx];

    // For simplified implementation, just return stored value
    if (node->isLeaf())
    {
        union { uint32_t i; float f; } converter;
        converter.i = node->getChildrenIndex();
        return converter.f;
    }

    // Traverse octree (simplified)
    while (!node->isLeaf())
    {
        const uint32_t childIdx = ((fracPart.z >= 0.5f) ? 4 : 0) +
                                  ((fracPart.y >= 0.5f) ? 2 : 0) +
                                  ((fracPart.x >= 0.5f) ? 1 : 0);

        const uint32_t nextIdx = node->getChildrenIndex() + childIdx;
        if (nextIdx >= mOctreeData.size())
        {
            return mMinBorderValue;
        }

        node = &mOctreeData[nextIdx];
        fracPart = glm::fract(2.0f * fracPart);
    }

    union { uint32_t i; float f; } converter;
    converter.i = node->getChildrenIndex();
    return converter.f;
}

float OctreeSdf::getDistance(glm::vec3 sample, glm::vec3& outGradient) const
{
    float dist = getDistance(sample);
    
    // Approximate gradient using finite differences
    const float eps = 1e-4f;
    glm::vec3 dx(eps, 0.0f, 0.0f);
    glm::vec3 dy(0.0f, eps, 0.0f);
    glm::vec3 dz(0.0f, 0.0f, eps);

    outGradient.x = (getDistance(sample + dx) - getDistance(sample - dx)) / (2.0f * eps);
    outGradient.y = (getDistance(sample + dy) - getDistance(sample - dy)) / (2.0f * eps);
    outGradient.z = (getDistance(sample + dz) - getDistance(sample - dz)) / (2.0f * eps);

    float len = glm::length(outGradient);
    if (len > 1e-6f)
    {
        outGradient /= len;
    }

    return dist;
}

void OctreeSdf::computeMinBorderValue()
{
    // Compute minimum value at the border of the SDF domain
    mMinBorderValue = 0.0f;
    
    // Sample at corners and find minimum
    for (int i = 0; i < 8; i++)
    {
        glm::vec3 corner(
            (i & 1) ? mBox.max.x : mBox.min.x,
            (i & 2) ? mBox.max.y : mBox.min.y,
            (i & 4) ? mBox.max.z : mBox.min.z
        );
        float dist = getDistance(corner);
        mMinBorderValue = std::min(mMinBorderValue, dist);
    }
}

void OctreeSdf::getDepthDensity(std::vector<float>& depthsDensity)
{
    depthsDensity.resize(mMaxDepth + 1);
    std::fill(depthsDensity.begin(), depthsDensity.end(), 0.0f);
    
    // Simplified: count nodes per depth
    uint32_t numLeaves = 0;
    for (const auto& node : mOctreeData)
    {
        if (node.isLeaf())
        {
            numLeaves++;
        }
    }
    
    float size = 1.0f;
    for (uint32_t d = 0; d < depthsDensity.size(); d++)
    {
        depthsDensity[d] = size * static_cast<float>(numLeaves);
        size *= 0.125f;
    }
}

std::optional<OctreeSdf::TerminationRule> OctreeSdf::stringToTerminationRule(const std::string& str)
{
    if (str == "trapezoidal_rule")
        return TerminationRule::TRAPEZOIDAL_RULE;
    if (str == "simpsons_rule")
        return TerminationRule::SIMPSONS_RULE;
    if (str == "by_distance_rule")
        return TerminationRule::BY_DISTANCE_RULE;
    if (str == "none")
        return TerminationRule::NONE;
    return std::nullopt;
}

}
