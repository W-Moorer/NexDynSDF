#include "sdflib/utils/NagataPatch.h"
#include <iostream>

namespace sdflib
{
namespace NagataPatch
{

    // =========================================================================
    // Math Utilities
    // =========================================================================

    glm::vec3 computeCurvature(glm::vec3 d, glm::vec3 n0, glm::vec3 n1)
    {
        static const float angleTol = 0.9999984769f; // cos(0.1 deg)
        
        glm::vec3 v = 0.5f * (n0 + n1);
        glm::vec3 Deltav = 0.5f * (n0 - n1);
        
        float dv = glm::dot(d, v);
        float dDeltav = glm::dot(d, Deltav);
        float Deltac = glm::dot(n0, Deltav);
        
        float c = 1.0f - 2.0f * Deltac;
        
        if (std::abs(c) <= angleTol)
        {
            return (dDeltav / (1.0f - Deltac)) * v + (dv / Deltac) * Deltav;
        }
        else
        {
            return glm::vec3(0.0f);
        }
    }

    // =========================================================================
    // Quintic Blending Math
    // =========================================================================

    float smoothStepQuintic(float s)
    {
        return s * s * s * (s * (s * 6.0f - 15.0f) + 10.0f);
    }
    
    float smoothStepQuinticDeriv(float s)
    {
        float t = s - 1.0f;
        return 30.0f * s * s * t * t;
    }

    void computeEffectiveCoefficients(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v,
        BlendingResult& res)
    {
        // Distance parametrizations
        float d[3] = { v, 1.0f - u, u - v }; // Distances to edges 0, 1, 2
        
        // Derivatives of d w.r.t u and v
        const float dd_du[3] = { 0.0f, -1.0f, 1.0f };
        const float dd_dv[3] = { 1.0f, 0.0f, -1.0f };

        for(int i=0; i<3; ++i)
        {
            if(!enhance.edges[i].enabled)
            {
                // No blending, constant value
                res.c_eff[i] = patch.c_orig[i];
                res.dc_du[i] = glm::vec3(0.0f);
                res.dc_dv[i] = glm::vec3(0.0f);
            }
            else
            {
                float d0 = enhance.edges[i].d0;
                float inv_d0 = enhance.edges[i].inv_d0;
                float dist = d[i];
                
                if(dist <= 0.0f) // On the edge (or outside in specific way)
                {
                     res.c_eff[i] = enhance.edges[i].c_sharp;
                     res.dc_du[i] = glm::vec3(0.0f);
                     res.dc_dv[i] = glm::vec3(0.0f);
                }
                else if(dist >= d0) // Far from edge
                {
                     res.c_eff[i] = patch.c_orig[i];
                     res.dc_du[i] = glm::vec3(0.0f);
                     res.dc_dv[i] = glm::vec3(0.0f);
                }
                else // Blending region
                {
                    float s = dist * inv_d0;
                    float w = smoothStepQuintic(s);
                    float dw_ds = smoothStepQuinticDeriv(s);
                    
                    glm::vec3 diff = patch.c_orig[i] - enhance.edges[i].c_sharp;
                    res.c_eff[i] = enhance.edges[i].c_sharp + w * diff;
                    
                    glm::vec3 factor = diff * dw_ds * inv_d0;
                    res.dc_du[i] = factor * dd_du[i];
                    res.dc_dv[i] = factor * dd_dv[i];
                }
            }
        }
    }

    // =========================================================================
    // Core Evaluation Functions
    // =========================================================================

    glm::vec3 evaluateSurface(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v)
    {
        BlendingResult blend;
        computeEffectiveCoefficients(patch, enhance, u, v, blend);

        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];

        float oneMinusU = 1.0f - u;
        float uMinusV = u - v;
        
        // Linear part
        glm::vec3 P_lin = x00 * oneMinusU + x10 * uMinusV + x11 * v;
        
        // Quadratic part
        glm::vec3 Q1 = blend.c_eff[0] * (oneMinusU * uMinusV);
        glm::vec3 Q2 = blend.c_eff[1] * (uMinusV * v);
        glm::vec3 Q3 = blend.c_eff[2] * (oneMinusU * v);
        
        return P_lin - Q1 - Q2 - Q3;
    }

    void evaluateDerivatives(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v,
        glm::vec3& dXdu, glm::vec3& dXdv)
    {
        BlendingResult blend;
        computeEffectiveCoefficients(patch, enhance, u, v, blend);

        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];

        // --- Linear Part Derivatives ---
        glm::vec3 dLin_du = -x00 + x10;
        glm::vec3 dLin_dv = -x10 + x11;

        // --- Quadratic Part Derivatives ---
        auto computeTermDerivs = [&](int idx, float b, float db_du, float db_dv, glm::vec3& dq_du, glm::vec3& dq_dv)
        {
            dq_du = blend.dc_du[idx] * b + blend.c_eff[idx] * db_du;
            dq_dv = blend.dc_dv[idx] * b + blend.c_eff[idx] * db_dv;
        };

        glm::vec3 dQ1_du, dQ1_dv;
        glm::vec3 dQ2_du, dQ2_dv;
        glm::vec3 dQ3_du, dQ3_dv;

        // Term 1: Basis b1 = (1-u)(u-v)
        float b1 = (1.0f - u) * (u - v);
        float db1_du = 1.0f - 2.0f * u + v;
        float db1_dv = u - 1.0f;
        computeTermDerivs(0, b1, db1_du, db1_dv, dQ1_du, dQ1_dv);

        // Term 2: Basis b2 = (u-v)v
        float b2 = (u - v) * v;
        float db2_du = v;
        float db2_dv = u - 2.0f * v;
        computeTermDerivs(1, b2, db2_du, db2_dv, dQ2_du, dQ2_dv);

        // Term 3: Basis b3 = (1-u)v
        float b3 = (1.0f - u) * v;
        float db3_du = -v;
        float db3_dv = 1.0f - u;
        computeTermDerivs(2, b3, db3_du, db3_dv, dQ3_du, dQ3_dv);

        dXdu = dLin_du - (dQ1_du + dQ2_du + dQ3_du);
        dXdv = dLin_dv - (dQ1_dv + dQ2_dv + dQ3_dv);
    }
    
    // =========================================================================
    // Projection (Newton-Raphson)
    // =========================================================================

    ProjectionResult findNearestPoint(
        glm::vec3 point, 
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        int maxIterations)
    {
        // Multi-start seeds to avoid local minima
        // Vertices: (0,0), (1,0), (1,1)
        // Midpoints: (0.5,0), (1,0.5), (0.5,0.5)
        // Centroid: (0.333, 0.333)
        static const std::array<std::pair<float, float>, 7> seeds = {{
            {0.333333f, 0.333333f},
            {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
            {0.5f, 0.0f}, {1.0f, 0.5f}, {0.5f, 0.5f}
        }};

        float bestU = 0.0f;
        float bestV = 0.0f;
        float minDistSq = std::numeric_limits<float>::max();
        glm::vec3 bestPoint;

        for (const auto& seed : seeds)
        {
            float u = seed.first;
            float v = seed.second;

            // Newton Iterations
            for(int i=0; i<maxIterations; ++i)
            {
                glm::vec3 S = evaluateSurface(patch, enhance, u, v);
                glm::vec3 diffVec = S - point; 
                
                glm::vec3 Su, Sv;
                evaluateDerivatives(patch, enhance, u, v, Su, Sv);
                
                // Objective F = 0.5 * |S - P|^2
                // Grad F = (S - P) * J
                float gradU = glm::dot(diffVec, Su);
                float gradV = glm::dot(diffVec, Sv);
                
                // Hessian Approx J^T * J
                float H_uu = glm::dot(Su, Su);
                float H_uv = glm::dot(Su, Sv);
                float H_vv = glm::dot(Sv, Sv);
                
                float det = H_uu * H_vv - H_uv * H_uv;
                if(std::abs(det) < 1e-10f) 
                {
                    break;
                }
                
                float invDet = 1.0f / det;
                float du = (H_vv * (-gradU) - H_uv * (-gradV)) * invDet;
                float dv = (-H_uv * (-gradU) + H_uu * (-gradV)) * invDet;
                
                float next_u = u + du;
                float next_v = v + dv;
                
                // Clamp to triangle domain u in [0,1], v in [0, u]
                // Note: v <= u is the diagonal.
                
                // Simple clamping strategy
                if (next_u < 0.0f) next_u = 0.0f;
                if (next_u > 1.0f) next_u = 1.0f;
                
                if (next_v < 0.0f) next_v = 0.0f;
                if (next_v > next_u) next_v = next_u; // Enforce v <= u
                
                if(std::abs(next_u - u) < 1e-5f && std::abs(next_v - v) < 1e-5f)
                {
                    u = next_u;
                    v = next_v;
                    break;
                }
                
                u = next_u;
                v = next_v;
            }
            
            glm::vec3 finalP = evaluateSurface(patch, enhance, u, v);
            float d2 = glm::dot(finalP - point, finalP - point);
            
            if (d2 < minDistSq)
            {
                minDistSq = d2;
                bestU = u;
                bestV = v;
                bestPoint = finalP;
            }
        }

        ProjectionResult result;
        result.nearestPoint = bestPoint;
        result.parameter = glm::vec3(bestU, bestV, 0.0f);
        result.sqDistance = minDistSq;
        
        return result;
    }
    
    // Phase 3: Smart initialization with custom hint
    ProjectionResult findNearestPointWithHint(
        glm::vec3 point, 
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        glm::vec2 hint,
        int maxIterations)
    {
        // Prioritize the hint, then try standard seeds
        std::array<std::pair<float, float>, 8> seeds;
        seeds[0] = {hint.x, hint.y};  // User hint first!
        seeds[1] = {0.333333f, 0.333333f};  // Centroid
        seeds[2] = {0.0f, 0.0f};
        seeds[3] = {1.0f, 0.0f};
        seeds[4] = {1.0f, 1.0f};
        seeds[5] = {0.5f, 0.0f};
        seeds[6] = {1.0f, 0.5f};
        seeds[7] = {0.5f, 0.5f};

        float bestU = 0.0f;
        float bestV = 0.0f;
        float minDistSq = std::numeric_limits<float>::max();
        glm::vec3 bestPoint;

        for (const auto& seed : seeds)
        {
            float u = seed.first;
            float v = seed.second;

            // Newton Iterations (same logic as findNearestPoint)
            for(int i=0; i<maxIterations; ++i)
            {
                glm::vec3 S = evaluateSurface(patch, enhance, u, v);
                glm::vec3 diffVec = S - point; 
                
                glm::vec3 Su, Sv;
                evaluateDerivatives(patch, enhance, u, v, Su, Sv);
                
                float gradU = glm::dot(diffVec, Su);
                float gradV = glm::dot(diffVec, Sv);
                
                float H_uu = glm::dot(Su, Su);
                float H_uv = glm::dot(Su, Sv);
                float H_vv = glm::dot(Sv, Sv);
                
                float det = H_uu * H_vv - H_uv * H_uv;
                if(std::abs(det) < 1e-10f) 
                {
                    break;
                }
                
                float invDet = 1.0f / det;
                float du = (H_vv * (-gradU) - H_uv * (-gradV)) * invDet;
                float dv = (-H_uv * (-gradU) + H_uu * (-gradV)) * invDet;
                
                float next_u = u + du;
                float next_v = v + dv;
                
                // Clamp to triangle domain
                if (next_u < 0.0f) next_u = 0.0f;
                if (next_u > 1.0f) next_u = 1.0f;
                
                if (next_v < 0.0f) next_v = 0.0f;
                if (next_v > next_u) next_v = next_u;
                
                if(std::abs(next_u - u) < 1e-5f && std::abs(next_v - v) < 1e-5f)
                {
                    u = next_u;
                    v = next_v;
                    break;
                }
                
                u = next_u;
                v = next_v;
            }
            
            glm::vec3 finalP = evaluateSurface(patch, enhance, u, v);
            float d2 = glm::dot(finalP - point, finalP - point);
            
            if (d2 < minDistSq)
            {
                minDistSq = d2;
                bestU = u;
                bestV = v;
                bestPoint = finalP;
            }
        }

        ProjectionResult result;
        result.nearestPoint = bestPoint;
        result.parameter = glm::vec3(bestU, bestV, 0.0f);
        result.sqDistance = minDistSq;
        
        return result;
    }

    float getSignedDistPointAndNagataPatch(
        const glm::vec3& point, 
        const NagataPatchData& patch,
        glm::vec3* outNearestPoint)
    {
        PatchEnhancementData dummyEnhance;
        ProjectionResult res = findNearestPoint(point, patch, dummyEnhance);
        
        if (outNearestPoint)
        {
            *outNearestPoint = res.nearestPoint;
        }
        
        glm::vec3 Su, Sv;
        evaluateDerivatives(patch, dummyEnhance, res.parameter.x, res.parameter.y, Su, Sv);
        
        glm::vec3 normal = glm::cross(Su, Sv);
        if (glm::dot(normal, normal) > 1e-12f)
        {
            normal = glm::normalize(normal);
        }
        else
        {
             // Fallback: use triangle normal?
             glm::vec3 e1 = patch.vertices[1] - patch.vertices[0];
             glm::vec3 e2 = patch.vertices[2] - patch.vertices[0];
             normal = glm::normalize(glm::cross(e1, e2));
        }

        glm::vec3 diff = point - res.nearestPoint;
        float sign = glm::dot(diff, normal) >= 0.0f ? 1.0f : -1.0f;
        
        return sign * std::sqrt(res.sqDistance);
    }
}
}
