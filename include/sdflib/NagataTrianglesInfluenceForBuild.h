#ifndef NAGATA_TRIANGLES_INFLUENCE_FOR_BUILD_H
#define NAGATA_TRIANGLES_INFLUENCE_FOR_BUILD_H

#include "utils/Mesh.h"
#include "utils/TriangleUtils.h"
#include "utils/NagataPatch.h"
#include "utils/GJK.h"
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
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <memory>
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
    bool mHasCreaseEdges = false;
    
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

        for (const auto& e : mEnhancedData)
        {
            if (e.edges[0].enabled || e.edges[1].enabled || e.edges[2].enabled)
            {
                mHasCreaseEdges = true;
                break;
            }
        }
    }

    struct EdgeAdjCache
    {
        std::vector<std::array<uint32_t, 3>> neighbors;
    };

    inline static uint64_t edgeKey(uint32_t a, uint32_t b)
    {
        const uint32_t lo = glm::min(a, b);
        const uint32_t hi = glm::max(a, b);
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    }

    inline static std::shared_ptr<EdgeAdjCache> getOrBuildAdjCache(const Mesh& mesh)
    {
        static std::mutex mtx;
        static std::unordered_map<const Mesh*, std::weak_ptr<EdgeAdjCache>> cache;

        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(&mesh);
        if (it != cache.end())
        {
            if (auto sp = it->second.lock()) return sp;
        }

        auto sp = std::make_shared<EdgeAdjCache>();
        const auto& indices = mesh.getIndices();
        const uint32_t triCount = static_cast<uint32_t>(indices.size() / 3u);
        sp->neighbors.assign(triCount, {std::numeric_limits<uint32_t>::max(),
                                        std::numeric_limits<uint32_t>::max(),
                                        std::numeric_limits<uint32_t>::max()});

        std::unordered_map<uint64_t, std::pair<uint32_t, uint8_t>> first;
        first.reserve(triCount * 3u);

        for (uint32_t t = 0; t < triCount; ++t)
        {
            const uint32_t i0 = indices[3u * t];
            const uint32_t i1 = indices[3u * t + 1u];
            const uint32_t i2 = indices[3u * t + 2u];

            const std::array<std::pair<uint32_t, uint32_t>, 3> edges = {
                std::make_pair(i0, i1),
                std::make_pair(i1, i2),
                std::make_pair(i0, i2)};

            for (uint8_t e = 0; e < 3u; ++e)
            {
                const uint64_t k = edgeKey(edges[e].first, edges[e].second);
                auto jt = first.find(k);
                if (jt == first.end())
                {
                    first.emplace(k, std::make_pair(t, e));
                    continue;
                }

                const uint32_t ot = jt->second.first;
                const uint8_t oe = jt->second.second;
                if (ot == t) continue;

                if (sp->neighbors[t][e] == std::numeric_limits<uint32_t>::max() &&
                    sp->neighbors[ot][oe] == std::numeric_limits<uint32_t>::max())
                {
                    sp->neighbors[t][e] = ot;
                    sp->neighbors[ot][oe] = t;
                }
            }
        }

        cache[&mesh] = sp;
        return sp;
    }

    inline bool tryComputeCreaseSignNormal(
        const Mesh& mesh,
        const EdgeAdjCache& adj,
        uint32_t triIdx,
        float u,
        float v,
        glm::vec3& outNormal) const
    {
        const float uvEps = 1.0e-5f;
        const float d0 = v;
        const float d1 = 1.0f - u;
        const float d2 = u - v;

        uint32_t edge = 0;
        float dmin = d0;
        if (d1 < dmin) { dmin = d1; edge = 1; }
        if (d2 < dmin) { dmin = d2; edge = 2; }

        if (dmin > uvEps) return false;
        if (!mEnhancedData[triIdx].edges[edge].enabled) return false;
        if (triIdx >= adj.neighbors.size()) return false;

        const uint32_t nbTri = adj.neighbors[triIdx][edge];
        if (nbTri == std::numeric_limits<uint32_t>::max()) return false;
        if (nbTri >= mPatches.size()) return false;

        uint32_t aLocal = 0, bLocal = 1;
        float t = 0.0f;
        if (edge == 0u)
        {
            aLocal = 0u; bLocal = 1u; t = u;
        }
        else if (edge == 1u)
        {
            aLocal = 1u; bLocal = 2u; t = v;
        }
        else
        {
            aLocal = 0u; bLocal = 2u; t = u;
        }

        t = glm::clamp(t, 0.0f, 1.0f);

        const auto& indices = mesh.getIndices();
        const uint32_t base = 3u * triIdx;
        const uint32_t nbBase = 3u * nbTri;
        if (base + 2u >= indices.size()) return false;
        if (nbBase + 2u >= indices.size()) return false;

        const uint32_t g0 = indices[base];
        const uint32_t g1 = indices[base + 1u];
        const uint32_t g2 = indices[base + 2u];
        const std::array<uint32_t, 3> gv = {g0, g1, g2};
        const uint32_t ga = gv[aLocal];
        const uint32_t gb = gv[bLocal];

        auto findLocal = [&](uint32_t tri, uint32_t g) -> int
        {
            const uint32_t b = 3u * tri;
            if (indices[b] == g) return 0;
            if (indices[b + 1u] == g) return 1;
            if (indices[b + 2u] == g) return 2;
            return -1;
        };

        const int aNbLocal = findLocal(nbTri, ga);
        const int bNbLocal = findLocal(nbTri, gb);
        if (aNbLocal < 0 || bNbLocal < 0) return false;

        glm::vec3 nA_L = mPatches[triIdx].normals[aLocal];
        glm::vec3 nB_L = mPatches[triIdx].normals[bLocal];
        glm::vec3 nA_R = mPatches[nbTri].normals[static_cast<uint32_t>(aNbLocal)];
        glm::vec3 nB_R = mPatches[nbTri].normals[static_cast<uint32_t>(bNbLocal)];

        auto safeNormalize = [](glm::vec3 n) -> glm::vec3
        {
            const float lenSq = glm::dot(n, n);
            if (std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z) && lenSq > 1.0e-12f)
            {
                return n / std::sqrt(lenSq);
            }
            return glm::vec3(0.0f);
        };

        nA_L = safeNormalize(nA_L);
        nB_L = safeNormalize(nB_L);
        nA_R = safeNormalize(nA_R);
        nB_R = safeNormalize(nB_R);
        if (glm::dot(nA_L, nA_L) <= 0.0f || glm::dot(nB_L, nB_L) <= 0.0f ||
            glm::dot(nA_R, nA_R) <= 0.0f || glm::dot(nB_R, nB_R) <= 0.0f)
        {
            return false;
        }

        if (glm::dot(nA_L, nA_R) < 0.0f) nA_R = -nA_R;
        if (glm::dot(nB_L, nB_R) < 0.0f) nB_R = -nB_R;

        const glm::vec3 nA = safeNormalize((1.0f - t) * nA_L + t * nA_R);
        const glm::vec3 nB = safeNormalize((1.0f - t) * nB_L + t * nB_R);
        const glm::vec3 sum = nA + nB;
        const glm::vec3 nRef = safeNormalize(sum);
        if (glm::dot(nRef, nRef) <= 0.0f) return false;

        outNormal = nRef;
        return true;
    }

    inline bool tryComputeCreaseRefNormalFromLinearOwner(
        const Mesh& mesh,
        const EdgeAdjCache& adj,
        uint32_t triIdx,
        uint8_t edge,
        float t,
        glm::vec3& outNormal) const
    {
        if (edge >= 3u) return false;
        if (!mEnhancedData[triIdx].edges[edge].enabled) return false;
        if (triIdx >= adj.neighbors.size()) return false;

        const uint32_t nbTri = adj.neighbors[triIdx][edge];
        if (nbTri == std::numeric_limits<uint32_t>::max()) return false;
        if (nbTri >= mPatches.size()) return false;

        uint32_t aLocal = 0u, bLocal = 1u;
        if (edge == 0u)
        {
            aLocal = 0u; bLocal = 1u;
        }
        else if (edge == 1u)
        {
            aLocal = 1u; bLocal = 2u;
        }
        else
        {
            aLocal = 0u; bLocal = 2u;
        }

        t = glm::clamp(t, 0.0f, 1.0f);

        const auto& indices = mesh.getIndices();
        const uint32_t base = 3u * triIdx;
        const uint32_t nbBase = 3u * nbTri;
        if (base + 2u >= indices.size()) return false;
        if (nbBase + 2u >= indices.size()) return false;

        const uint32_t g0 = indices[base];
        const uint32_t g1 = indices[base + 1u];
        const uint32_t g2 = indices[base + 2u];
        const std::array<uint32_t, 3> gv = {g0, g1, g2};
        const uint32_t ga = gv[aLocal];
        const uint32_t gb = gv[bLocal];

        auto findLocal = [&](uint32_t tri, uint32_t g) -> int
        {
            const uint32_t b = 3u * tri;
            if (indices[b] == g) return 0;
            if (indices[b + 1u] == g) return 1;
            if (indices[b + 2u] == g) return 2;
            return -1;
        };

        const int aNbLocal = findLocal(nbTri, ga);
        const int bNbLocal = findLocal(nbTri, gb);
        if (aNbLocal < 0 || bNbLocal < 0) return false;

        glm::vec3 nA_L = mPatches[triIdx].normals[aLocal];
        glm::vec3 nB_L = mPatches[triIdx].normals[bLocal];
        glm::vec3 nA_R = mPatches[nbTri].normals[static_cast<uint32_t>(aNbLocal)];
        glm::vec3 nB_R = mPatches[nbTri].normals[static_cast<uint32_t>(bNbLocal)];

        auto safeNormalize = [](glm::vec3 n) -> glm::vec3
        {
            const float lenSq = glm::dot(n, n);
            if (std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z) && lenSq > 1.0e-12f)
            {
                return n / std::sqrt(lenSq);
            }
            return glm::vec3(0.0f);
        };

        nA_L = safeNormalize(nA_L);
        nB_L = safeNormalize(nB_L);
        nA_R = safeNormalize(nA_R);
        nB_R = safeNormalize(nB_R);
        if (glm::dot(nA_L, nA_L) <= 0.0f || glm::dot(nB_L, nB_L) <= 0.0f ||
            glm::dot(nA_R, nA_R) <= 0.0f || glm::dot(nB_R, nB_R) <= 0.0f)
        {
            return false;
        }

        if (glm::dot(nA_L, nA_R) < 0.0f) nA_R = -nA_R;
        if (glm::dot(nB_L, nB_R) < 0.0f) nB_R = -nB_R;

        const glm::vec3 nA = safeNormalize((1.0f - t) * nA_L + t * nA_R);
        const glm::vec3 nB = safeNormalize((1.0f - t) * nB_L + t * nB_R);
        const glm::vec3 sum = nA + nB;
        const glm::vec3 nRef = safeNormalize(sum);
        if (glm::dot(nRef, nRef) <= 0.0f) return false;

        outNormal = nRef;
        return true;
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

        const auto& indices = mesh.getIndices();
        const auto& vertices = mesh.getVertices();

        std::unordered_map<uint32_t, std::vector<uint32_t>> vertexToTriangles;
        std::unordered_map<uint64_t, std::vector<uint32_t>> edgeToTriangles;
        vertexToTriangles.reserve(triangles.size() * 3u);
        edgeToTriangles.reserve(triangles.size() * 3u);
        for (uint32_t t : triangles)
        {
            const uint32_t base = 3u * t;
            if (base + 2u >= indices.size()) continue;
            const uint32_t i0 = indices[base];
            const uint32_t i1 = indices[base + 1u];
            const uint32_t i2 = indices[base + 2u];
            vertexToTriangles[i0].push_back(t);
            vertexToTriangles[i1].push_back(t);
            vertexToTriangles[i2].push_back(t);
            edgeToTriangles[edgeKey(i0, i1)].push_back(t);
            edgeToTriangles[edgeKey(i1, i2)].push_back(t);
            edgeToTriangles[edgeKey(i0, i2)].push_back(t);
        }

        std::shared_ptr<EdgeAdjCache> adjCache;
        if (mHasCreaseEdges)
        {
            adjCache = getOrBuildAdjCache(mesh);
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

            uint32_t triOwnerLin = 0u;
            float ownerMinDistSq = FLT_MAX;
            for (uint32_t t : triangles)
            {
                const float distSq = TriangleUtils::getSqDistPointAndTriangle(inPoints[i], trianglesData[t]);
                if (distSq < ownerMinDistSq)
                {
                    ownerMinDistSq = distSq;
                    triOwnerLin = t;
                }
            }

            const TriangleUtils::LinearClosestInfo ownerInfo =
                TriangleUtils::getLinearClosestInfoPointAndTriangle(inPoints[i], trianglesData[triOwnerLin]);

            std::vector<uint32_t> nagataCandidates = candidates[i];
            if (std::find(nagataCandidates.begin(), nagataCandidates.end(), triOwnerLin) == nagataCandidates.end())
            {
                nagataCandidates.push_back(triOwnerLin);
            }

            const uint32_t ownerBase = 3u * triOwnerLin;
            if (ownerBase + 2u < indices.size())
            {
                const uint32_t g0 = indices[ownerBase];
                const uint32_t g1 = indices[ownerBase + 1u];
                const uint32_t g2 = indices[ownerBase + 2u];

                if (ownerInfo.entityType == TriangleUtils::ClosestEntityType::Vertex)
                {
                    const uint32_t gv = (ownerInfo.entityLocalId == 0u) ? g0 : (ownerInfo.entityLocalId == 1u ? g1 : g2);
                    auto it = vertexToTriangles.find(gv);
                    if (it != vertexToTriangles.end())
                    {
                        nagataCandidates.insert(nagataCandidates.end(), it->second.begin(), it->second.end());
                    }
                }
                else if (ownerInfo.entityType == TriangleUtils::ClosestEntityType::Edge)
                {
                    uint32_t ga = g0, gb = g1;
                    if (ownerInfo.entityLocalId == 0u)
                    {
                        ga = g0; gb = g1;
                    }
                    else if (ownerInfo.entityLocalId == 1u)
                    {
                        ga = g1; gb = g2;
                    }
                    else
                    {
                        ga = g0; gb = g2;
                    }

                    auto it = edgeToTriangles.find(edgeKey(ga, gb));
                    if (it != edgeToTriangles.end())
                    {
                        nagataCandidates.insert(nagataCandidates.end(), it->second.begin(), it->second.end());
                    }
                }
            }

            std::sort(nagataCandidates.begin(), nagataCandidates.end());
            nagataCandidates.erase(std::unique(nagataCandidates.begin(), nagataCandidates.end()), nagataCandidates.end());

            // Find best candidate using Nagata projection
            float minExactDistSq = 9.0e8f;
            float secondExactDistSq = 9.0e8f;
            
            uint32_t bestTriIdx = 0;
            glm::vec3 bestNearestPt(0.0f);
            float bestU = 0.0f, bestV = 0.0f;
            bool foundValid = false;

            for(uint32_t t : nagataCandidates)
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
                    secondExactDistSq = minExactDistSq;
                    minExactDistSq = d2;
                    bestTriIdx = t;
                    bestNearestPt = result.nearestPoint;
                    bestU = result.parameter.x;
                    bestV = result.parameter.y;
                    foundValid = true;
                }
                else if (d2 < secondExactDistSq)
                {
                    secondExactDistSq = d2;
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

            float dotVal = glm::dot(diff, surfNormal);
            float sign = (dotVal < 0.0f) ? -1.0f : 1.0f;
            
            glm::vec3 finalGradient;

            if (dist > 1e-6f)
            {
                // Gradient = sign * normalize(diff)
                finalGradient = sign * (diff / dist);
            }
            else
            {
                finalGradient = sign * nagataNormal;
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
                SPDLOG_INFO("NagataDebug vtx i={} proj tri={} u={:.6f} v={:.6f} dist={:.6f} dot={:.6f} sign={} surfN=({:.6f},{:.6f},{:.6f}) diff=({:.6f},{:.6f},{:.6f})",
                            i,
                            bestTriIdx,
                            bestU, bestV,
                            dist,
                            dotVal,
                            sign,
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
        (void)gjkIter;
        (void)gjkCallsInside;
        (void)gjkIterHistogramInside;
        (void)gjkCallsOutside;
        (void)gjkIterHistogramOutside;
    }
};

} // namespace sdflib

#endif // NAGATA_TRIANGLES_INFLUENCE_FOR_BUILD_H
