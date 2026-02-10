#ifndef NAGATA_TRIANGLES_INFLUENCE_FOR_BUILD_H
#define NAGATA_TRIANGLES_INFLUENCE_FOR_BUILD_H

#include "utils/Mesh.h"
#include "utils/TriangleUtils.h"
#include "utils/NagataPatch.h"
#include "utils/GJK.h"
#include "utils/WindingNumberOracle.h"
#include "InterpolationMethods.h"

#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <algorithm>
#include <cfloat>
#include <limits>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstdint>
#include <spdlog/spdlog.h>

namespace sdflib
{

/**
 * @brief TrianglesInfluence strategy using Nagata patches for vertex SDF calculation
 * 
 * This strategy uses Nagata curved surfaces to compute accurate vertex SDF values
 * during octree construction, eliminating geometric error from flat approximation.
 * Based on NagataTrianglesInfluence but uses simpler PatchEnhancementData.
 */
template<typename T>
struct NagataTrianglesInfluenceForBuild
{
    typedef T InterpolationMethod;
    typedef float VertexInfo;  // Stores minimum squared distance
    struct NodeInfo {};
    
    // FULL-FIELD NAGATA STRATEGY:
    // All vertices use Nagata projection for C0 continuity
    // No narrowband threshold - unified distance metric across entire field
    
    // Nagata data (passed at construction)
    std::vector<NagataPatch::NagataPatchData> mPatches;
    std::vector<NagataPatch::PatchEnhancementData> mEnhancedData;
    
    // Statistics (for debugging)
    uint32_t gjkIter = 0;
    uint32_t gjkCallsInside = 0;
    std::array<uint32_t, 10> gjkIterHistogramInside = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t gjkCallsOutside = 0;
    std::array<uint32_t, 10> gjkIterHistogramOutside = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    struct DebugConfig
    {
        bool enabled = false;
        glm::vec3 point = glm::vec3(0.0f);
        float radius = 0.0f;
        int maxLogs = 200;
    };

    inline static std::atomic<int>& debugLogCount()
    {
        static std::atomic<int> counter{0};
        return counter;
    }

    inline static DebugConfig& debugConfig()
    {
        static DebugConfig cfg = []() -> DebugConfig
        {
            DebugConfig out;

            const char* p = std::getenv("NEXDYN_SDF_DEBUG_POINT");
            if (!p || !*p) return out;

            std::string s(p);
            for (char& c : s) if (c == ',') c = ' ';

            std::istringstream iss(s);
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (!(iss >> x >> y >> z)) return out;

            out.enabled = true;
            out.point = glm::vec3(x, y, z);

            if (const char* r = std::getenv("NEXDYN_SDF_DEBUG_RADIUS"))
            {
                std::istringstream rss(r);
                float rv = 0.0f;
                if (rss >> rv) out.radius = rv;
            }
            if (out.radius <= 0.0f) out.radius = 0.1f;

            if (const char* m = std::getenv("NEXDYN_SDF_DEBUG_MAX_LOGS"))
            {
                std::istringstream mss(m);
                int mv = 0;
                if (mss >> mv) out.maxLogs = mv;
            }
            if (out.maxLogs <= 0) out.maxLogs = 200;

            SPDLOG_INFO("NagataDebug enabled point=({:.6f},{:.6f},{:.6f}) radius={} maxLogs={}",
                        out.point.x, out.point.y, out.point.z, out.radius, out.maxLogs);

            return out;
        }();

        return cfg;
    }

    inline static bool rayParityInsideWithDir(glm::vec3 point, const Mesh& mesh, const glm::vec3& dir, float salt)
    {
        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        const std::vector<uint32_t>& indices = mesh.getIndices();

        glm::vec3 d = glm::normalize(dir);

        float h = std::sin(point.x * 12.9898f + point.y * 78.233f + point.z * 37.719f + salt) * 43758.5453f;
        float frac = h - std::floor(h);
        glm::vec3 jitter = glm::normalize(glm::vec3(0.123f, 0.456f, 0.789f) + frac * glm::vec3(0.31f, 0.17f, 0.23f));
        glm::vec3 o = point + 1.0e-4f * jitter;

        uint32_t hits = 0;
        const float eps = 1.0e-8f;

        const size_t triCount = indices.size() / 3;
        for (size_t t = 0; t < triCount; t++)
        {
            const glm::vec3 a = vertices[indices[3 * t]];
            const glm::vec3 b = vertices[indices[3 * t + 1]];
            const glm::vec3 c = vertices[indices[3 * t + 2]];

            const glm::vec3 e1 = b - a;
            const glm::vec3 e2 = c - a;
            const glm::vec3 pvec = glm::cross(d, e2);
            const float det = glm::dot(e1, pvec);
            if (std::abs(det) <= eps) continue;

            const float invDet = 1.0f / det;
            const glm::vec3 tvec = o - a;
            const float u = glm::dot(tvec, pvec) * invDet;
            if (u < 0.0f || u > 1.0f) continue;

            const glm::vec3 qvec = glm::cross(tvec, e1);
            const float v = glm::dot(d, qvec) * invDet;
            if (v < 0.0f || (u + v) > 1.0f) continue;

            const float tHit = glm::dot(e2, qvec) * invDet;
            if (tHit > eps) hits++;
        }

        return (hits & 1u) == 1u;
    }

    inline static bool rayParityInside(glm::vec3 point, const Mesh& mesh)
    {
        return rayParityInsideWithDir(point, mesh, glm::vec3(0.852f, 0.231f, 0.468f), 0.0f);
    }

    inline static std::shared_ptr<const WindingNumberOracle> windingOracleForMesh(const Mesh& mesh)
    {
        const auto vPtr = reinterpret_cast<std::uintptr_t>(mesh.getVertices().data());
        const auto iPtr = reinterpret_cast<std::uintptr_t>(mesh.getIndices().data());
        const uint64_t key =
            static_cast<uint64_t>(vPtr) ^
            (static_cast<uint64_t>(iPtr) << 1) ^
            (static_cast<uint64_t>(mesh.getVertices().size()) * 0x9e3779b97f4a7c15ULL) ^
            (static_cast<uint64_t>(mesh.getIndices().size()) * 0xbf58476d1ce4e5b9ULL);

        thread_local uint64_t tlKey = 0;
        thread_local std::shared_ptr<const WindingNumberOracle> tlOracle;
        if (tlOracle && tlKey == key)
        {
            return tlOracle;
        }

        static std::mutex m;
        static std::unordered_map<uint64_t, std::weak_ptr<const WindingNumberOracle>> cache;

        std::lock_guard<std::mutex> lock(m);
        auto it = cache.find(key);
        if (it != cache.end())
        {
            if (auto alive = it->second.lock())
            {
                tlKey = key;
                tlOracle = alive;
                return alive;
            }
        }

        WindingNumberOracle::Settings settings;
        settings.theta = 0.25;
        settings.leafMaxTriangles = 8;

        try
        {
            auto created = std::make_shared<WindingNumberOracle>(mesh.getVertices(), mesh.getIndices(), settings);
            cache[key] = created;
            tlKey = key;
            tlOracle = created;
            return created;
        }
        catch (...)
        {
            cache.erase(key);
            return nullptr;
        }
    }

    inline static bool robustInside(glm::vec3 point, const Mesh& mesh)
    {
        const bool a = rayParityInsideWithDir(point, mesh, glm::vec3(0.852f, 0.231f, 0.468f), 0.0f);
        const bool b = rayParityInsideWithDir(point, mesh, glm::vec3(-0.217f, 0.931f, 0.293f), 19.19f);
        const bool c = rayParityInsideWithDir(point, mesh, glm::vec3(0.381f, -0.128f, 0.916f), 37.37f);

        if ((a == b) && (b == c))
        {
            return a;
        }

        const auto oracle = windingOracleForMesh(mesh);
        if (oracle)
        {
            return oracle->inside(point);
        }

        const int votes = static_cast<int>(a) + static_cast<int>(b) + static_cast<int>(c);
        return votes >= 2;
    }
    
    /**
     * @brief Constructor
     * @param patches Nagata patch data for all triangles
     * @param enhanced Enhanced edge data for crease handling
     */
    NagataTrianglesInfluenceForBuild(
        const std::vector<NagataPatch::NagataPatchData>& patches,
        const std::vector<NagataPatch::PatchEnhancementData>& enhanced)
        : mPatches(patches), mEnhancedData(enhanced)
    {
        if (patches.size() != enhanced.size())
        {
            throw std::runtime_error("Nagata patches and enhancement data size mismatch");
        }
    }
    
    /**
     * @brief Calculate vertex SDF values using Nagata patches
     * Simplified version based on NagataTrianglesInfluence
     */
    template<size_t N>
    inline void calculateVerticesInfo(
        const glm::vec3 nodeCenter, 
        const float nodeHalfSize,
        const std::vector<uint32_t>& triangles,
        const std::array<glm::vec3, N>& pointsRelPos,
        const uint32_t pointsToInterpolateMask,
        const std::array<float, InterpolationMethod::NUM_COEFFICIENTS>& interpolationCoeff,
        std::array<std::array<float, InterpolationMethod::VALUES_PER_VERTEX>, N>& outPointsValues,
        std::array<VertexInfo, N>& outPointsInfo,
        const Mesh& mesh, 
        const std::vector<TriangleUtils::TriangleData>& trianglesData)
    {
        // 1. Calculate world positions
        std::array<glm::vec3, N> inPoints;
        for(size_t i=0; i < N; i++)
        {
            inPoints[i] = nodeCenter + pointsRelPos[i] * nodeHalfSize;
        }

        // 2. Initialize info
        outPointsInfo.fill(std::numeric_limits<float>::max());
        
        // Temporary storage for candidates
        std::array<std::vector<uint32_t>, N> candidates;
        std::array<float, N> minFlatDist;
        minFlatDist.fill(std::numeric_limits<float>::max());
        
        // Pass 1: Find global minimum flat distance and collect potential candidates
        for(uint32_t t : triangles)
        {
            float defl = mPatches[t].maxDeflection;

            for(size_t i=0; i < N; i++)
            {
                if(pointsToInterpolateMask & (1 << (N-i-1))) continue;

                float distSq = TriangleUtils::getSqDistPointAndTriangle(inPoints[i], trianglesData[t]);
                float dist = std::sqrt(distSq);
                
                if(dist < minFlatDist[i]) minFlatDist[i] = dist;
                
                // Check against running minimum
                if(dist - defl < minFlatDist[i] + 1e-4f) 
                {
                    candidates[i].push_back(t);
                }
            }
        }

        // Pass 2: Strict Filter against Final Minimum (with dynamic margin)
        for(size_t i=0; i < N; i++)
        {
            if(pointsToInterpolateMask & (1 << (N-i-1)))
            {
                // CRITICAL: Handle interpolated points correctly
                InterpolationMethod::interpolateVertexValues(
                    interpolationCoeff, 0.5f * pointsRelPos[i] + 0.5f, 2.0f * nodeHalfSize, outPointsValues[i]);
                outPointsInfo[i] = std::abs(outPointsValues[i][0]); 
                continue;
            }
            
            size_t writeIdx = 0;
            for(size_t readIdx = 0; readIdx < candidates[i].size(); readIdx++)
            {
                uint32_t t = candidates[i][readIdx];
                float dist = std::sqrt(TriangleUtils::getSqDistPointAndTriangle(inPoints[i], trianglesData[t]));
                
                // PHASE 2: Dynamic margin for far-field points
                // Far-field maxDeflection is too small, causing empty candidate lists
                float margin = (minFlatDist[i] < 0.5f) 
                    ? mPatches[t].maxDeflection 
                    : mPatches[t].maxDeflection * 3.0f;  // 3x margin for far-field
                
                if (dist - margin < minFlatDist[i] + 1e-4f)
                {
                    candidates[i][writeIdx++] = t;
                }
            }
            candidates[i].resize(writeIdx);
        }

        // 3. Narrow Phase - FULL-FIELD NAGATA STRATEGY
        // All vertices use Nagata projection for C0 continuity
        for(size_t i=0; i < N; i++)
        {
            if(pointsToInterpolateMask & (1 << (N-i-1)))
            {
                // Interpolated points already handled in Pass 2
                continue; 
            }

            const DebugConfig& dbg = debugConfig();
            const bool dbgNear = dbg.enabled && (glm::length(inPoints[i] - dbg.point) <= dbg.radius);
            const bool dbgAllowLog = dbgNear && (debugLogCount().load(std::memory_order_relaxed) < dbg.maxLogs);
            if (dbgAllowLog)
            {
                int n = debugLogCount().fetch_add(1, std::memory_order_relaxed);
                if (n < dbg.maxLogs)
                {
                    SPDLOG_INFO("NagataDebug vtx i={} p=({:.6f},{:.6f},{:.6f}) nodeC=({:.6f},{:.6f},{:.6f}) hs={:.6f} tris={} minFlat={:.6f} cand={}",
                                i,
                                inPoints[i].x, inPoints[i].y, inPoints[i].z,
                                nodeCenter.x, nodeCenter.y, nodeCenter.z,
                                nodeHalfSize,
                                triangles.size(),
                                minFlatDist[i],
                                candidates[i].size());
                }
            }

            // Fallback: if no candidates after filtering, use nearest flat triangle
            if (candidates[i].empty())
            {
                // Narrowband fallback: Use flat triangle as last resort
                uint32_t nearestTriIdx = 0;
                float minDistSq = FLT_MAX;
                for (uint32_t t : triangles)
                {
                    float distSq = TriangleUtils::getSqDistPointAndTriangle(inPoints[i], trianglesData[t]);
                    if (distSq < minDistSq)
                    {
                        minDistSq = distSq;
                        nearestTriIdx = t;
                    }
                }
                
                glm::vec3 gradient;
                float signedDist = TriangleUtils::getSignedDistPointAndTriangle(
                    inPoints[i], trianglesData[nearestTriIdx], gradient);
                
                if (dbgAllowLog)
                {
                    SPDLOG_INFO("NagataDebug vtx i={} fallback=emptyCand tri={} flatDist={:.6f} signDist={:.6f} grad=({:.6f},{:.6f},{:.6f})",
                                i, nearestTriIdx, std::sqrt(minDistSq), signedDist, gradient.x, gradient.y, gradient.z);
                }

                outPointsValues[i][0] = signedDist;
                outPointsValues[i][1] = gradient.x;
                outPointsValues[i][2] = gradient.y;
                outPointsValues[i][3] = gradient.z;
                
                for (size_t j = 4; j < InterpolationMethod::VALUES_PER_VERTEX; j++)
                {
                    outPointsValues[i][j] = 0.0f;
                }
                
                outPointsInfo[i] = std::sqrt(minDistSq);
                continue;
            }

            // Find best candidate using Nagata projection
            float minExactDistSq = 9.0e8f;
            
            uint32_t bestTriIdx = 0;
            glm::vec3 bestNearestPt(0.0f);
            float bestU = 0.0f, bestV = 0.0f;
            bool foundValid = false;

            for(uint32_t t : candidates[i])
            {
                // PHASE 3: Smart Initialization
                // Use flat triangle projection as initial guess for Newton solver
                
                // Step 1: Project point onto flat triangle
                const auto& tri = trianglesData[t];
                glm::vec3 localPt = tri.transform * (inPoints[i] - tri.origin);
                
                // Step 2: Convert to barycentric (u,v)
                // Triangle vertices in parameter space: (0,0), (1,0), (1,1)
                // clamp to triangle
                float u_hint = glm::clamp(localPt.x / tri.v2, 0.0f, 1.0f);
                float v_hint = glm::clamp(localPt.y / tri.v3.y, 0.0f, 1.0f);
                
                // Enforce barycentric constraint: v <= u
                if (v_hint > u_hint) {
                    float s = u_hint / (u_hint + v_hint);
                    u_hint = u_hint * s;
                    v_hint = v_hint * s;
                }
                
                // Step 3: Use hint for Newton iteration
                NagataPatch::ProjectionResult result = NagataPatch::findNearestPointWithHint(
                    inPoints[i], mPatches[t], mEnhancedData[t], glm::vec2(u_hint, v_hint), 8);
                
                float d2 = result.sqDistance;
                
                // Sanitize d2
                if (std::isnan(d2) || std::isinf(d2)) continue;

                if (d2 < minExactDistSq)
                {
                    minExactDistSq = d2;
                    bestTriIdx = t;
                    bestNearestPt = result.nearestPoint;
                    bestU = result.parameter.x;
                    bestV = result.parameter.y;
                    foundValid = true;
                }
            }
            
            if (!foundValid)
            {
                // CRITICAL FIX: Newton iteration failed for all candidates
                // Fall back to flat triangle distance (same as empty candidates case)
                
                uint32_t nearestTriIdx = 0;
                float minDistSq = FLT_MAX;
                for (uint32_t t : triangles)
                {
                    float distSq = TriangleUtils::getSqDistPointAndTriangle(inPoints[i], trianglesData[t]);
                    if (distSq < minDistSq)
                    {
                        minDistSq = distSq;
                        nearestTriIdx = t;
                    }
                }
                
                glm::vec3 gradient;
                float signedDist = TriangleUtils::getSignedDistPointAndTriangle(
                    inPoints[i], trianglesData[nearestTriIdx], gradient);
                
                if (dbgAllowLog)
                {
                    SPDLOG_INFO("NagataDebug vtx i={} fallback=noValidProj tri={} flatDist={:.6f} signDist={:.6f} grad=({:.6f},{:.6f},{:.6f})",
                                i, nearestTriIdx, std::sqrt(minDistSq), signedDist, gradient.x, gradient.y, gradient.z);
                }

                outPointsValues[i][0] = signedDist;
                outPointsValues[i][1] = gradient.x;
                outPointsValues[i][2] = gradient.y;
                outPointsValues[i][3] = gradient.z;
                
                for (size_t j = 4; j < InterpolationMethod::VALUES_PER_VERTEX; j++)
                {
                    outPointsValues[i][j] = 0.0f;
                }
                
                outPointsInfo[i] = std::sqrt(minDistSq);
                continue;
            }

            outPointsInfo[i] = std::sqrt(minExactDistSq);

            // Gradient Calculation
            glm::vec3 diff = inPoints[i] - bestNearestPt;
            float dist = std::sqrt(minExactDistSq);
            
            // Compute surface normal at projection point
            glm::vec3 dXdu, dXdv;
            NagataPatch::evaluateDerivatives(
                mPatches[bestTriIdx], 
                mEnhancedData[bestTriIdx],
                bestU, bestV,
                dXdu, dXdv);
            
            glm::vec3 surfNormal = glm::normalize(glm::cross(dXdu, dXdv));
            {
                const glm::vec3 ref = mPatches[bestTriIdx].normals[0] +
                                      mPatches[bestTriIdx].normals[1] +
                                      mPatches[bestTriIdx].normals[2];
                if (!glm::isnan(ref.x + ref.y + ref.z))
                {
                    const float refLenSq = glm::dot(ref, ref);
                    if (refLenSq > 1.0e-12f && glm::dot(surfNormal, ref) < 0.0f)
                    {
                        surfNormal = -surfNormal;
                    }
                }
            }

            float dotVal = glm::dot(diff, surfNormal);
            float sign = (dotVal < 0.0f) ? -1.0f : 1.0f;

            const float uvEps = 1.0e-5f;
            const bool uvDegenerate = (bestU <= uvEps) || (bestV <= uvEps) || ((bestU - bestV) <= uvEps) ||
                                      (bestU >= (1.0f - uvEps)) || (bestV >= (1.0f - uvEps));
            const bool normalDegenerate = std::isnan(surfNormal.x + surfNormal.y + surfNormal.z) || (glm::length(surfNormal) <= 1.0e-6f);
            const float dotEps = 1.0e-5f * glm::max(dist, 1.0f);
            const bool dotDegenerate = std::abs(dotVal) <= dotEps;
            const bool needOracle = uvDegenerate || normalDegenerate || dotDegenerate;

            if (needOracle)
            {
                const bool inside = robustInside(inPoints[i], mesh);
                sign = inside ? -1.0f : 1.0f;
            }
            
            glm::vec3 finalGradient;

            if (dist > 1e-6f)
            {
                // Gradient = sign * normalize(diff)
                finalGradient = sign * (diff / dist);
            }
            else
            {
                // On surface, use surface normal
                finalGradient = surfNormal;
            }

            // Sanitize Gradient
            if (std::isnan(finalGradient.x) || std::isnan(finalGradient.y) || std::isnan(finalGradient.z))
            {
                finalGradient = (glm::length(surfNormal) > 1e-6f) ? surfNormal : glm::vec3(1,0,0);
            }

            // Final Assignment
            float signedDist = sign * dist;

            if (dbgAllowLog)
            {
                SPDLOG_INFO("NagataDebug vtx i={} proj tri={} u={:.6f} v={:.6f} dist={:.6f} dot={:.6f} sign={} oracle={} surfN=({:.6f},{:.6f},{:.6f}) diff=({:.6f},{:.6f},{:.6f})",
                            i,
                            bestTriIdx,
                            bestU, bestV,
                            dist,
                            dotVal,
                            sign,
                            needOracle ? 1 : 0,
                            surfNormal.x, surfNormal.y, surfNormal.z,
                            diff.x, diff.y, diff.z);
            }
            
            // Sanitize Distance
            if (std::isnan(signedDist) || std::isinf(signedDist))
            {
                 signedDist = 3.0e4f;
            }

            outPointsValues[i][0] = signedDist;
            outPointsValues[i][1] = finalGradient.x;
            outPointsValues[i][2] = finalGradient.y;
            outPointsValues[i][3] = finalGradient.z;
            
            // Clear remaining fields
            for (size_t j = 4; j < InterpolationMethod::VALUES_PER_VERTEX; j++)
            {
                outPointsValues[i][j] = 0.0f;
            }
        }
    }
    
    /**
     * @brief Filter triangles using GJK (same as BasicTrianglesInfluence)
     */
    inline void filterTriangles(
        const glm::vec3 nodeCenter, 
        const float nodeHalfSize,
        const std::vector<uint32_t>& inTriangles, 
        std::vector<uint32_t>& outTriangles,
        const std::array<std::array<float, InterpolationMethod::VALUES_PER_VERTEX>, 8>& verticesValues,
        const std::array<VertexInfo, 8>& verticesInfo,
        const Mesh& mesh, 
        const std::vector<TriangleUtils::TriangleData>& trianglesData)
    {
        outTriangles.clear();
        
        float maxMinDist = 0.0f;
        for(uint32_t i=0; i < 8; i++) maxMinDist = glm::max(maxMinDist, verticesInfo[i]);

        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        const std::vector<uint32_t>& indices = mesh.getIndices();
        std::array<glm::vec3, 3> triangle;

        for(const uint32_t& idx : inTriangles)
        {
            triangle[0] = vertices[indices[3 * idx]] - nodeCenter;
            triangle[1] = vertices[indices[3 * idx + 1]] - nodeCenter;
            triangle[2] = vertices[indices[3 * idx + 2]] - nodeCenter;

            uint32_t iter = 0;
            if(GJK::IsNearMinimize(glm::vec3(nodeHalfSize), triangle, maxMinDist, &iter))
            {
                gjkIterHistogramInside[glm::min(iter, 9u)]++;
                gjkCallsInside++;
                gjkIter += iter;
                outTriangles.push_back(idx);
            }
            else
            {
                gjkIterHistogramOutside[glm::min(iter, 9u)]++;
                gjkCallsOutside++;
                gjkIter += iter;
            }
        }
    }
    
    void printStatistics()
    {
        // Optional: Add logging if needed
    }
};

} // namespace sdflib

#endif // NAGATA_TRIANGLES_INFLUENCE_FOR_BUILD_H
