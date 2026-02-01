/**
 * @file OctreeSdf.h
 * @brief Adaptive octree-based SDF with trilinear interpolation
 */

#pragma once

#include "SdfFunction.h"
#include "utils/Mesh.h"
#include <vector>
#include <array>
#include <functional>
#include <optional>

namespace sdflib
{

struct OctreeNode
{
    uint32_t childrenIndex = 0;
    
    bool isLeaf() const { return (childrenIndex & 0x80000000) != 0; }
    uint32_t getChildrenIndex() const { return childrenIndex & 0x7FFFFFFF; }
    void setChildrenIndex(uint32_t index) { childrenIndex = index; }
    void setLeaf() { childrenIndex |= 0x80000000; }
    void removeMark() { childrenIndex &= 0x7FFFFFFF; }
    
    template<class Archive>
    void serialize(Archive& archive)
    {
        archive(childrenIndex);
    }
};

class OctreeSdf : public SdfFunction
{
public:
    enum class InitAlgorithm
    {
        UNIFORM,
        NO_CONTINUITY,
        CONTINUITY
    };

    enum class TerminationRule
    {
        NONE,
        TRAPEZOIDAL_RULE,
        SIMPSONS_RULE,
        BY_DISTANCE_RULE
    };

    struct TerminationRuleParams
    {
        float threshold = 1e-3f;
        float thresholdByDistance = 0.0f;

        static TerminationRuleParams setNoneRuleParams() 
        { 
            return TerminationRuleParams(); 
        }
        
        static TerminationRuleParams setTrapezoidalRuleParams(float threshold)
        {
            TerminationRuleParams params;
            params.threshold = threshold;
            return params;
        }
        
        static TerminationRuleParams setByDistanceRuleParams(float threshold, float thresholdByDistance)
        {
            TerminationRuleParams params;
            params.threshold = threshold;
            params.thresholdByDistance = thresholdByDistance;
            return params;
        }

        template<class Archive>
        void serialize(Archive& archive)
        {
            archive(threshold, thresholdByDistance);
        }
    };

    OctreeSdf() = default;
    
    OctreeSdf(const Mesh& mesh, BoundingBox box, 
              uint32_t depth, uint32_t startDepth = 1,
              float terminationThreshold = 1e-3f,
              InitAlgorithm initAlgorithm = InitAlgorithm::CONTINUITY,
              uint32_t numThreads = 1);
    
    OctreeSdf(const Mesh& mesh, BoundingBox box, 
              uint32_t depth, uint32_t startDepth,
              TerminationRule terminationRule, 
              TerminationRuleParams params,
              InitAlgorithm initAlgorithm = InitAlgorithm::CONTINUITY,
              uint32_t numThreads = 1);

    float getDistance(glm::vec3 sample) const override;
    float getDistance(glm::vec3 sample, glm::vec3& outGradient) const override;
    BoundingBox getBoundingBox() const override { return mBox; }
    SdfFormat getFormat() const override { return SdfFormat::OCTREE; }

    void getDepthDensity(std::vector<float>& depthsDensity);
    
    static std::optional<TerminationRule> stringToTerminationRule(const std::string& str);

    template<class Archive>
    void serialize(Archive& archive)
    {
        archive(mBox, mMaxDepth, mStartGridSize, mStartGridXY, 
                mStartGridCellSize, mMinBorderValue, mOctreeData);
    }

private:
    BoundingBox mBox;
    uint32_t mMaxDepth = 0;
    uint32_t mStartGridSize = 0;
    uint32_t mStartGridXY = 0;
    float mStartGridCellSize = 0.0f;
    float mMinBorderValue = 0.0f;
    std::vector<OctreeNode> mOctreeData;

    void buildOctree(const Mesh& mesh, BoundingBox box, 
                     uint32_t depth, uint32_t startDepth,
                     TerminationRule terminationRule, 
                     TerminationRuleParams params,
                     InitAlgorithm initAlgorithm, 
                     uint32_t numThreads);

    void initUniformOctree(const Mesh& mesh, uint32_t startDepth, uint32_t depth);
    
    template<typename DistanceQuery>
    void initOctree(const Mesh& mesh, uint32_t startDepth, uint32_t depth,
                    TerminationRule terminationRule, 
                    TerminationRuleParams params,
                    uint32_t numThreads);

    template<typename DistanceQuery>
    void initOctreeWithContinuity(const Mesh& mesh, uint32_t startDepth, uint32_t depth,
                                  TerminationRule terminationRule, 
                                  TerminationRuleParams params);

    template<typename DistanceQuery>
    void initOctreeWithContinuityNoDelay(const Mesh& mesh, uint32_t startDepth, uint32_t depth,
                                         TerminationRule terminationRule, 
                                         TerminationRuleParams params,
                                         uint32_t numThreads);

    void computeMinBorderValue();
};

}
