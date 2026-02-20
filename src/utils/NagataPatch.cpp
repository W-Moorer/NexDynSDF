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

    static float gaussianDecay(float d, float k)
    {
        return std::exp(-k * d * d);
    }

    static float gaussianDecayDeriv(float d, float k)
    {
        return -2.0f * k * d * std::exp(-k * d * d);
    }

    static bool anyEdgeEnabled(const PatchEnhancementData& enhance)
    {
        return enhance.edges[0].enabled || enhance.edges[1].enabled || enhance.edges[2].enabled;
    }

    static bool computeReferenceNormal(
        const NagataPatchData& patch,
        glm::vec3& n_ref)
    {
        glm::vec3 triNormal = glm::cross(patch.vertices[1] - patch.vertices[0], patch.vertices[2] - patch.vertices[0]);
        float len = glm::length(triNormal);
        if (len > 1e-12f)
        {
            n_ref = triNormal / len;
            return true;
        }

        glm::vec3 avg = patch.normals[0] + patch.normals[1] + patch.normals[2];
        len = glm::length(avg);
        if (len > 1e-12f)
        {
            n_ref = avg / len;
            return true;
        }

        return false;
    }

    static glm::vec3 evaluateSurfaceOriginal(
        const NagataPatchData& patch,
        float u, float v)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];

        float oneMinusU = 1.0f - u;
        float uMinusV = u - v;

        glm::vec3 P_lin = x00 * oneMinusU + x10 * uMinusV + x11 * v;
        glm::vec3 Q1 = patch.c_orig[0] * (oneMinusU * uMinusV);
        glm::vec3 Q2 = patch.c_orig[1] * (uMinusV * v);
        glm::vec3 Q3 = patch.c_orig[2] * (oneMinusU * v);

        return P_lin - Q1 - Q2 - Q3;
    }

    static glm::vec3 applyEdgeCrossingGuard(
        const NagataPatchData& patch,
        const PatchEnhancementData& enhance,
        float u, float v,
        const glm::vec3& n_ref,
        glm::vec3 point)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];

        const std::array<int, 3> enabled = {
            enhance.edges[0].enabled ? 1 : 0,
            enhance.edges[1].enabled ? 1 : 0,
            enhance.edges[2].enabled ? 1 : 0
        };

        if (!enabled[0] && !enabled[1] && !enabled[2])
        {
            return point;
        }

        struct EdgeInfo
        {
            glm::vec3 a;
            glm::vec3 b;
            glm::vec3 opp;
            glm::vec3 c_sharp;
            float t;
            bool enabled;
        };

        const std::array<EdgeInfo, 3> edges = {{
            {x00, x10, x11, enhance.edges[0].c_sharp, u, enhance.edges[0].enabled},
            {x10, x11, x00, enhance.edges[1].c_sharp, v, enhance.edges[1].enabled},
            {x00, x11, x10, enhance.edges[2].c_sharp, v, enhance.edges[2].enabled}
        }};

        for (const auto& edge : edges)
        {
            if (!edge.enabled)
            {
                continue;
            }

            glm::vec3 e = edge.b - edge.a;
            glm::vec3 sideDir = glm::cross(n_ref, e);
            float sideLen = glm::length(sideDir);
            if (sideLen < 1e-12f)
            {
                continue;
            }
            sideDir /= sideLen;

            if (glm::dot(sideDir, edge.opp - (edge.a + edge.b) * 0.5f) < 0.0f)
            {
                sideDir = -sideDir;
            }

            float t = edge.t;
            glm::vec3 edgePoint = (1.0f - t) * edge.a + t * edge.b - edge.c_sharp * t * (1.0f - t);
            float s = glm::dot(point - edgePoint, sideDir);
            if (s < 0.0f)
            {
                point -= s * sideDir;
            }
        }

        return point;
    }

    void computeEffectiveCoefficients(
        const NagataPatchData& patch, 
        const PatchEnhancementData& enhance,
        float u, float v,
        BlendingResult& res)
    {
        float d[3] = { v, 1.0f - u, u - v };
        const float dd_du[3] = { 0.0f, -1.0f, 1.0f };
        const float dd_dv[3] = { 1.0f, 0.0f, -1.0f };

        for(int i=0; i<3; ++i)
        {
            if(!enhance.edges[i].enabled)
            {
                res.c_eff[i] = patch.c_orig[i];
                res.dc_du[i] = glm::vec3(0.0f);
                res.dc_dv[i] = glm::vec3(0.0f);
                continue;
            }

            float k = enhance.edges[i].k_factor;
            float dist = d[i];
            float w = gaussianDecay(dist, k);
            glm::vec3 delta = enhance.edges[i].c_sharp - patch.c_orig[i];

            res.c_eff[i] = patch.c_orig[i] + delta * w;

            float dw_dd = gaussianDecayDeriv(dist, k);
            glm::vec3 dcd = delta * dw_dd;
            res.dc_du[i] = dcd * dd_du[i];
            res.dc_dv[i] = dcd * dd_dv[i];
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
        
        glm::vec3 point = P_lin - Q1 - Q2 - Q3;

        if (!anyEdgeEnabled(enhance))
        {
            return point;
        }

        glm::vec3 n_ref;
        if (!computeReferenceNormal(patch, n_ref))
        {
            return point;
        }

        glm::vec3 dXdu, dXdv;
        evaluateDerivatives(patch, enhance, u, v, dXdu, dXdv);
        float jac = glm::dot(glm::cross(dXdu, dXdv), n_ref);
        if (jac <= 0.0f)
        {
            return evaluateSurfaceOriginal(patch, u, v);
        }

        return applyEdgeCrossingGuard(patch, enhance, u, v, n_ref, point);
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
