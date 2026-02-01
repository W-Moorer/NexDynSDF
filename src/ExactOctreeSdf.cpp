/**
 * @file ExactOctreeSdf.cpp
 * @brief Exact distance query octree SDF implementation
 */

#include "sdflib/ExactOctreeSdf.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace sdflib
{

ExactOctreeSdf::ExactOctreeSdf(const Mesh& mesh, BoundingBox box,
                               uint32_t depth, uint32_t startDepth,
                               uint32_t minTrianglesPerNode,
                               uint32_t numThreads)
{
    buildOctree(mesh, box, depth, startDepth, minTrianglesPerNode, numThreads);
}

void ExactOctreeSdf::buildOctree(const Mesh& mesh, BoundingBox box,
                                 uint32_t depth, uint32_t startDepth,
                                 uint32_t minTrianglesPerNode,
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

    SPDLOG_INFO("Building exact octree with depth {}, start depth {}", depth, startDepth);

    // Store mesh data for exact queries
    mVertices.reserve(mesh.getVertices().size() * 3);
    for (const auto& v : mesh.getVertices())
    {
        mVertices.push_back(v.x);
        mVertices.push_back(v.y);
        mVertices.push_back(v.z);
    }

    mIndices = mesh.getIndices();

    // Initialize distance query
    initDistanceQuery();

    // Initialize uniform grid
    const uint32_t numStartCells = mStartGridSize * mStartGridSize * mStartGridSize;
    mOctreeData.resize(numStartCells);
    mLeafData.resize(numStartCells);

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

                // Query exact distance
                tmd::Result result = mDistanceQuery->signed_distance(
                    {cellCenter.x, cellCenter.y, cellCenter.z}
                );

                // Store as leaf node
                mOctreeData[idx].setLeaf();
                mOctreeData[idx].setChildrenIndex(idx);

                mLeafData[idx].distance = static_cast<float>(result.distance);
                mLeafData[idx].gradient = glm::vec3(
                    result.nearest_point.v[0] - cellCenter.x,
                    result.nearest_point.v[1] - cellCenter.y,
                    result.nearest_point.v[2] - cellCenter.z
                );
                float len = glm::length(mLeafData[idx].gradient);
                if (len > 1e-6f)
                {
                    mLeafData[idx].gradient /= len;
                }
            }
        }
    }

    // Compute minimum border value
    mMinBorderValue = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        glm::vec3 corner(
            (i & 1) ? mBox.max.x : mBox.min.x,
            (i & 2) ? mBox.max.y : mBox.min.y,
            (i & 4) ? mBox.max.z : mBox.min.z
        );
        tmd::Result result = mDistanceQuery->signed_distance({corner.x, corner.y, corner.z});
        mMinBorderValue = std::min(mMinBorderValue, static_cast<float>(result.distance));
    }

    SPDLOG_INFO("Exact octree build complete. Nodes: {}", mOctreeData.size());
}

void ExactOctreeSdf::initDistanceQuery() const
{
    if (mDistanceQuery) return;

    std::vector<tmd::Vec3d> vertices;
    vertices.reserve(mVertices.size() / 3);
    for (size_t i = 0; i < mVertices.size(); i += 3)
    {
        vertices.push_back(tmd::Vec3d(mVertices[i], mVertices[i + 1], mVertices[i + 2]));
    }

    std::vector<std::array<int, 3>> triangles;
    triangles.reserve(mIndices.size() / 3);
    for (size_t i = 0; i < mIndices.size(); i += 3)
    {
        triangles.push_back({
            static_cast<int>(mIndices[i]),
            static_cast<int>(mIndices[i + 1]),
            static_cast<int>(mIndices[i + 2])
        });
    }

    mDistanceQuery = std::make_unique<tmd::TriangleMeshDistance>();
    mDistanceQuery->construct(vertices, triangles);
}

float ExactOctreeSdf::getDistance(glm::vec3 sample) const
{
    glm::vec3 fracPart = (sample - mBox.min) / mStartGridCellSize;
    glm::ivec3 startArrayPos = glm::floor(fracPart);

    if (startArrayPos.x < 0 || startArrayPos.x >= static_cast<int>(mStartGridSize) ||
        startArrayPos.y < 0 || startArrayPos.y >= static_cast<int>(mStartGridSize) ||
        startArrayPos.z < 0 || startArrayPos.z >= static_cast<int>(mStartGridSize))
    {
        // Outside domain - return exact distance
        initDistanceQuery();
        tmd::Result result = mDistanceQuery->signed_distance({sample.x, sample.y, sample.z});
        return static_cast<float>(result.distance);
    }

    const uint32_t idx = startArrayPos.z * mStartGridXY +
                         startArrayPos.y * mStartGridSize +
                         startArrayPos.x;

    if (idx >= mOctreeData.size())
    {
        initDistanceQuery();
        tmd::Result result = mDistanceQuery->signed_distance({sample.x, sample.y, sample.z});
        return static_cast<float>(result.distance);
    }

    // For this simplified implementation, use exact distance query
    initDistanceQuery();
    tmd::Result result = mDistanceQuery->signed_distance({sample.x, sample.y, sample.z});
    return static_cast<float>(result.distance);
}

float ExactOctreeSdf::getDistance(glm::vec3 sample, glm::vec3& outGradient) const
{
    initDistanceQuery();
    tmd::Result result = mDistanceQuery->signed_distance({sample.x, sample.y, sample.z});

    outGradient = glm::vec3(
        result.nearest_point.v[0] - sample.x,
        result.nearest_point.v[1] - sample.y,
        result.nearest_point.v[2] - sample.z
    );
    float len = glm::length(outGradient);
    if (len > 1e-6f)
    {
        outGradient /= len;
    }

    return static_cast<float>(result.distance);
}

}
