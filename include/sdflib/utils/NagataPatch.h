#ifndef NAGATA_PATCH_H
#define NAGATA_PATCH_H

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

namespace sdflib
{
namespace NagataPatch
{
    // =========================================================================
    // Core Data Structures
    // =========================================================================

    /**
     * @brief Standard computes curve coefficient (Nagata 2005)
     */
    glm::vec3 computeCurvature(glm::vec3 d, glm::vec3 n0, glm::vec3 n1);

    /**
     * @brief Data for a single Nagata Patch (geometric definition)
     */
    struct NagataPatchData
    {
        glm::vec3 vertices[3];      // x00, x10, x11
        glm::vec3 normals[3];       // n0, n1, n2 (used for degenerate fallback / initial guess)
        glm::vec3 c_orig[3];        // Original quadratic coefficients for edges 0, 1, 2
        
        // Edge Indices mapping for looking up shared data (optional, used by higher level)
        int32_t edgeIndices[3] = {-1, -1, -1};
        
        // Conservative bound of surface deviation from flat triangle
        float maxDeflection = 0.0f;
        
        NagataPatchData() = default;
        
        NagataPatchData(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                        const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2)
        {
            vertices[0] = v0; vertices[1] = v1; vertices[2] = v2;
            normals[0] = n0; normals[1] = n1; normals[2] = n2;
            
            c_orig[0] = computeCurvature(v1 - v0, n0, n1);
            c_orig[1] = computeCurvature(v2 - v1, n1, n2);
            c_orig[2] = computeCurvature(v2 - v0, n0, n2);
            
            // Calculate max deflection based on curvature coefficients
            // A safe bound is the max magnitude of any control coefficient
            // This represents how far the curve might deviate from the flat triangle
            maxDeflection = 0.0f;
            maxDeflection = std::max(maxDeflection, glm::length(c_orig[0]));
            maxDeflection = std::max(maxDeflection, glm::length(c_orig[1]));
            maxDeflection = std::max(maxDeflection, glm::length(c_orig[2]));
        }
    };

    /**
     * @brief Enhancement data for a single edge
     * Used to blend between original coefficient and sharp coefficient.
     */
    struct EdgeEnhancement
    {
        bool enabled = false;       // Is this edge a crease/crack requiring repair?
        glm::vec3 c_sharp;          // The shared coefficient for this edge
        float k_factor = 0.0f;
    };

    /**
     * @brief Container for the 3 edges' enhancement data of a patch
     */
    struct PatchEnhancementData
    {
        std::array<EdgeEnhancement, 3> edges;

        PatchEnhancementData() 
        {
            edges[0].enabled = false;
            edges[1].enabled = false;
            edges[2].enabled = false;
        }
    };

    struct BlendingResult
    {
        glm::vec3 c_eff[3];      // Effective coefficients for current (u,v)
        glm::vec3 dc_du[3];      // Partial derivatives of c_eff w.r.t u
        glm::vec3 dc_dv[3];      // Partial derivatives of c_eff w.r.t v
    };

    struct ProjectionResult
    {
        glm::vec3 nearestPoint;
        glm::vec3 parameter; // u, v, 0
        float sqDistance;
    };

    // =========================================================================
    // Functions
    // =========================================================================



    /**
     * @brief Calculates effective coefficients and their derivatives at (u,v)
     */
    void computeEffectiveCoefficients(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v,
        BlendingResult& res);

    /**
     * @brief Evaluate Surface Position
     */
    glm::vec3 evaluateSurface(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v);

    /**
     * @brief Evaluate Surface derivatives (Jacobian)
     */
    void evaluateDerivatives(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v,
        glm::vec3& dXdu, glm::vec3& dXdv);
    
    /**
     * @brief Find nearest point on surface using Newton's method
     */
    ProjectionResult findNearestPoint(
        glm::vec3 point, 
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        int maxIterations = 8);
    
    /**
     * @brief Find nearest point with custom initial hint (prioritized)
     * @param hint Initial guess (u,v) to try first before multi-start seeds
     */
    ProjectionResult findNearestPointWithHint(
        glm::vec3 point, 
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        glm::vec2 hint,
        int maxIterations = 8);

    float getSignedDistPointAndNagataPatch(
        const glm::vec3& point, 
        const NagataPatchData& patch,
        glm::vec3* outNearestPoint = nullptr);
}
}

#endif // NAGATA_PATCH_H
