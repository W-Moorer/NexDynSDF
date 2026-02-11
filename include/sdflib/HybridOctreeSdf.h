#ifndef HYBRID_OCTREE_SDF_H
#define HYBRID_OCTREE_SDF_H

#include "OctreeSdf.h"
#include "OctreeSdfBreadthFirst.h"
#include "NagataTrianglesInfluenceForBuild.h"
#include "utils/NagataPatch.h"
#include "utils/MeshBinaryLoader.h"

namespace sdflib
{

/**
 * @brief Hybrid OctreeSDF using Nagata patches for vertex value computation
 * 
 * This class builds an OctreeSDF where vertex SDF values are computed using
 * Nagata curved surfaces instead of flat triangles, eliminating geometric error
 * while maintaining fast query performance.
 */
class HybridOctreeSdf : public OctreeSdf
{
public:
    HybridOctreeSdf() : OctreeSdf() {}
    
    /**
     * @brief Construct hybrid SDF using Nagata patches
     * @param mesh Input mesh
     * @param patches Nagata patch data (one per triangle)
     * @param enhancedData Enhanced edge data for crease handling
     * @param box Bounding box
     * @param depth Maximum octree depth
     * @param startDepth Starting grid depth
     * @param terminationRule Subdivision termination rule
     * @param params Termination parameters
     * @param numThreads Number of threads (default=1)
     */
    HybridOctreeSdf(const Mesh& mesh,
                    const std::vector<NagataPatch::NagataPatchData>& patches,
                    const std::vector<NagataPatch::PatchEnhancementData>& enhancedData,
                    BoundingBox box, uint32_t depth, uint32_t startDepth,
                    TerminationRule terminationRule, TerminationRuleParams params,
                    uint32_t numThreads = 1)
    {
        // Create strategy with Nagata data
        NagataTrianglesInfluenceForBuild<TriCubicInterpolation> strategy(patches, enhancedData);
        
        // Setup basic members  (copied from OctreeSdf::buildOctree)
        mMaxDepth = depth;
        
        const glm::vec3 bbSize = box.getSize();
        const float maxSize = glm::max(glm::max(bbSize.x, bbSize.y), bbSize.z);
        mBox.min = box.getCenter() - 0.5f * maxSize;
        mBox.max = box.getCenter() + 0.5f * maxSize;
        
        mStartGridSize = 1 << startDepth;
        mStartGridXY = mStartGridSize * mStartGridSize;
        mStartGridCellSize = maxSize / static_cast<float>(mStartGridSize);
        
        // Build octree using Nagata strategy
        initOctreeWithContinuity(mesh, startDepth, depth, terminationRule, params, strategy);
        
        // Compute boundary values
        computeMinBorderValue();

        strategy.printStatistics();
    }
};

} // namespace sdflib

#endif // HYBRID_OCTREE_SDF_H
