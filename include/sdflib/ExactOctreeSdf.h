/**
 * @file ExactOctreeSdf.h
 * @brief Exact distance query octree SDF using TriangleMeshDistance
 */

#pragma once

#include "SdfFunction.h"
#include "utils/Mesh.h"
#include "TriangleMeshDistance.h"
#include <vector>
#include <memory>

namespace sdflib
{

class ExactOctreeSdf : public SdfFunction
{
public:
    ExactOctreeSdf() = default;
    
    ExactOctreeSdf(const Mesh& mesh, BoundingBox box,
                   uint32_t depth, uint32_t startDepth = 1,
                   uint32_t minTrianglesPerNode = 32,
                   uint32_t numThreads = 1);

    float getDistance(glm::vec3 sample) const override;
    float getDistance(glm::vec3 sample, glm::vec3& outGradient) const override;
    BoundingBox getBoundingBox() const override { return mBox; }
    SdfFormat getFormat() const override { return SdfFormat::EXACT_OCTREE; }

    template<class Archive>
    void serialize(Archive& archive)
    {
        archive(mBox, mMaxDepth, mStartGridSize, mStartGridXY,
                mStartGridCellSize, mMinBorderValue, mOctreeData,
                mVertices, mIndices);
    }

private:
    struct OctreeNode
    {
        uint32_t childrenIndex = 0;
        
        bool isLeaf() const { return (childrenIndex & 0x80000000) != 0; }
        uint32_t getChildrenIndex() const { return childrenIndex & 0x7FFFFFFF; }
        void setChildrenIndex(uint32_t index) { childrenIndex = index & 0x7FFFFFFF; }
        void setLeaf() { childrenIndex |= 0x80000000; }
        
        template<class Archive>
        void serialize(Archive& archive)
        {
            archive(childrenIndex);
        }
    };

    struct LeafData
    {
        float distance;
        glm::vec3 gradient;
        
        template<class Archive>
        void serialize(Archive& archive)
        {
            archive(distance, gradient);
        }
    };

    BoundingBox mBox;
    uint32_t mMaxDepth = 0;
    uint32_t mStartGridSize = 0;
    uint32_t mStartGridXY = 0;
    float mStartGridCellSize = 0.0f;
    float mMinBorderValue = 0.0f;
    
    std::vector<OctreeNode> mOctreeData;
    std::vector<LeafData> mLeafData;
    
    // For exact distance queries
    std::vector<float> mVertices;
    std::vector<uint32_t> mIndices;
    mutable std::unique_ptr<tmd::TriangleMeshDistance> mDistanceQuery;

    void buildOctree(const Mesh& mesh, BoundingBox box,
                     uint32_t depth, uint32_t startDepth,
                     uint32_t minTrianglesPerNode,
                     uint32_t numThreads);

    void initDistanceQuery() const;
};

}
