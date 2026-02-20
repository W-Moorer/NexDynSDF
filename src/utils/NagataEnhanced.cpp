#include "sdflib/utils/NagataEnhanced.h"
#include <cstring>
#include <algorithm>

namespace sdflib
{
namespace NagataEnhanced
{
    // ============================================================
    // ENG file I/O
    // ============================================================
    
    bool loadEnhancedData(const std::string& filepath, EnhancedNagataData& data)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }
        
        // Read header
        char magic[4];
        file.read(magic, 4);
        if (std::memcmp(magic, ENG_MAGIC, 4) != 0)
        {
            std::cerr << "NagataEnhanced: Invalid ENG file magic" << std::endl;
            return false;
        }
        
        uint32_t version, numEdges, reserved;
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&numEdges), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&reserved), sizeof(uint32_t));
        
        if (version != ENG_VERSION)
        {
            std::cerr << "NagataEnhanced: Unsupported ENG version " << version << std::endl;
            return false;
        }
        
        // Read data
        data.c_sharps.clear();
        for (uint32_t i = 0; i < numEdges; ++i)
        {
            uint32_t v0, v1;
            float cx, cy, cz;
            
            file.read(reinterpret_cast<char*>(&v0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&v1), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&cx), sizeof(float));
            file.read(reinterpret_cast<char*>(&cy), sizeof(float));
            file.read(reinterpret_cast<char*>(&cz), sizeof(float));
            
            if (!file)
            {
                std::cerr << "NagataEnhanced: ENG file data incomplete" << std::endl;
                return false;
            }
            
            data.c_sharps[EdgeKey(v0, v1)] = glm::vec3(cx, cy, cz);
        }
        
        std::cout << "NagataEnhanced: Loaded " << numEdges << " crease edges from " << filepath << std::endl;
        return true;
    }
    
    bool saveEnhancedData(const std::string& filepath, const EnhancedNagataData& data)
    {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "NagataEnhanced: Cannot create file " << filepath << std::endl;
            return false;
        }
        
        // Write header
        file.write(ENG_MAGIC, 4);
        uint32_t version = ENG_VERSION;
        uint32_t numEdges = static_cast<uint32_t>(data.c_sharps.size());
        uint32_t reserved = 0;
        
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&numEdges), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&reserved), sizeof(uint32_t));
        
        // Write data
        for (const auto& [key, c] : data.c_sharps)
        {
            uint32_t v0 = key.v0;
            uint32_t v1 = key.v1;
            float cx = c.x, cy = c.y, cz = c.z;
            
            file.write(reinterpret_cast<const char*>(&v0), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&v1), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&cx), sizeof(float));
            file.write(reinterpret_cast<const char*>(&cy), sizeof(float));
            file.write(reinterpret_cast<const char*>(&cz), sizeof(float));
        }
        
        std::cout << "NagataEnhanced: Saved " << numEdges << " crease edges to " << filepath << std::endl;
        return true;
    }
    
    std::string getEngFilepath(const std::string& nsmPath)
    {
        size_t dotPos = nsmPath.rfind('.');
        if (dotPos == std::string::npos)
        {
            return nsmPath + ".eng";
        }
        return nsmPath.substr(0, dotPos) + ".eng";
    }
    
    bool hasEngCache(const std::string& nsmPath)
    {
        std::ifstream file(getEngFilepath(nsmPath));
        return file.good();
    }
    
    // ============================================================
    // Crease edge detection
    // ============================================================
    
    std::map<EdgeKey, CreaseEdgeInfo> detectCreaseEdges(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        float gapThreshold)
    {
        // Edge to triangles mapping
        struct EdgeTriInfo
        {
            int triIdx;
            uint32_t v0, v1;
            int local0, local1;
        };
        
        std::map<EdgeKey, std::vector<EdgeTriInfo>> edgeToTris;
        
        for (size_t triIdx = 0; triIdx < faces.size(); ++triIdx)
        {
            const auto& tri = faces[triIdx];
            std::array<std::tuple<int, int, int, int>, 3> edges = {{
                {0, 1, 0, 1},
                {1, 2, 1, 2},
                {0, 2, 0, 2}
            }};
            
            for (const auto& [l0, l1, local0, local1] : edges)
            {
                EdgeKey key(tri[l0], tri[l1]);
                EdgeTriInfo info;
                info.triIdx = static_cast<int>(triIdx);
                info.v0 = tri[l0];
                info.v1 = tri[l1];
                info.local0 = local0;
                info.local1 = local1;
                edgeToTris[key].push_back(info);
            }
        }
        
        std::map<EdgeKey, CreaseEdgeInfo> creaseEdges;
        
        for (const auto& [edgeKey, trisInfo] : edgeToTris)
        {
            if (trisInfo.size() != 2)
                continue;  // Boundary edge or non-manifold
            
            const auto& triL = trisInfo[0];
            const auto& triR = trisInfo[1];
            
            glm::vec3 A = vertices[edgeKey.v0];
            glm::vec3 B = vertices[edgeKey.v1];
            
            auto getNormalAtVertex = [&](int triIdx, uint32_t globalVIdx, glm::vec3& outNormal) -> bool
            {
                const auto& tri = faces[triIdx];
                for (int i = 0; i < 3; ++i)
                {
                    if (tri[i] == globalVIdx)
                    {
                        outNormal = faceNormals[triIdx][i];
                        return true;
                    }
                }
                return false;
            };
            
            glm::vec3 n_A_L, n_B_L, n_A_R, n_B_R;
            if (!getNormalAtVertex(triL.triIdx, edgeKey.v0, n_A_L) ||
                !getNormalAtVertex(triL.triIdx, edgeKey.v1, n_B_L) ||
                !getNormalAtVertex(triR.triIdx, edgeKey.v0, n_A_R) ||
                !getNormalAtVertex(triR.triIdx, edgeKey.v1, n_B_R))
            {
                continue;
            }
            
            glm::vec3 e = B - A;
            glm::vec3 c_L = NagataPatch::computeCurvature(e, n_A_L, n_B_L);
            glm::vec3 c_R = NagataPatch::computeCurvature(e, n_A_R, n_B_R);
            
            // Sample for gap measurement
            float maxGap = 0.0f;
            for (int i = 0; i <= 10; ++i)
            {
                float t = static_cast<float>(i) / 10.0f;
                glm::vec3 p_L = (1.0f - t) * A + t * B - c_L * t * (1.0f - t);
                glm::vec3 p_R = (1.0f - t) * A + t * B - c_R * t * (1.0f - t);
                float gap = glm::length(p_L - p_R);
                maxGap = std::max(maxGap, gap);
            }
            
            if (maxGap > gapThreshold)
            {
                CreaseEdgeInfo info;
                info.A = A;
                info.B = B;
                info.n_A_L = n_A_L;
                info.n_A_R = n_A_R;
                info.n_B_L = n_B_L;
                info.n_B_R = n_B_R;
                info.tri_L = triL.triIdx;
                info.tri_R = triR.triIdx;
                info.max_gap = maxGap;
                creaseEdges[edgeKey] = info;
            }
        }
        
        return creaseEdges;
    }
    
    // ============================================================
    // c_sharp calculation
    // ============================================================
    
    glm::vec3 computeCreaseDirection(glm::vec3 nL, glm::vec3 nR, glm::vec3 e)
    {
        glm::vec3 d = glm::cross(nL, nR);
        float len = glm::length(d);
        
        if (len < 1e-8f)
        {
            return glm::normalize(e);
        }
        
        d = d / len;
        
        if (glm::dot(d, e) < 0.0f)
            d = -d;
        
        return d;
    }
    
    glm::vec3 computeCSharp(glm::vec3 A, glm::vec3 B, glm::vec3 dA, glm::vec3 dB)
    {
        glm::vec3 e = B - A;
        
        if (glm::dot(dA, dB) < 0.0f)
            dB = -dB;
        
        float G00 = glm::dot(dA, dA);
        float G01 = glm::dot(dA, dB);
        float G11 = glm::dot(dB, dB);
        float r0 = 2.0f * glm::dot(e, dA);
        float r1 = 2.0f * glm::dot(e, dB);
        
        float lambda = 1e-6f;
        G00 += lambda;
        G11 += lambda;
        
        float det = G00 * G11 - G01 * G01;
        float lA = 0.0f;
        float lB = 0.0f;
        if (std::abs(det) < 1e-12f)
        {
            float eLen = glm::length(e);
            lA = eLen;
            lB = eLen;
        }
        else
        {
            lA = (G11 * r0 - G01 * r1) / det;
            lB = (-G01 * r0 + G00 * r1) / det;
        }
        
        glm::vec3 T_A = lA * dA;
        glm::vec3 T_B = lB * dB;
        
        glm::vec3 c_sharp = 0.5f * (T_B - T_A);
        
        float eLen = glm::length(e);
        float cLen = glm::length(c_sharp);
        float maxC = 2.0f * eLen;
        if (cLen > maxC)
        {
            c_sharp = c_sharp * (maxC / cLen);
        }
        
        return c_sharp;
    }

    static int edgeIndexForTriangle(const std::array<uint32_t, 3>& tri, const EdgeKey& edgeKey)
    {
        if (EdgeKey(tri[0], tri[1]) == edgeKey) return 0;
        if (EdgeKey(tri[1], tri[2]) == edgeKey) return 1;
        if (EdgeKey(tri[0], tri[2]) == edgeKey) return 2;
        return -1;
    }

    static std::vector<std::pair<float, float>> sampleUvForEdge(int edgeIdx, int steps = 5, float eps = 0.05f)
    {
        std::vector<std::pair<float, float>> uvs;
        if (steps <= 0)
        {
            return uvs;
        }

        uvs.reserve(static_cast<size_t>(steps));
        float denom = (steps > 1) ? static_cast<float>(steps - 1) : 1.0f;
        for (int i = 0; i < steps; ++i)
        {
            float t = eps + (1.0f - 2.0f * eps) * (static_cast<float>(i) / denom);
            if (edgeIdx == 0)
            {
                uvs.emplace_back(t, eps);
            }
            else if (edgeIdx == 1)
            {
                uvs.emplace_back(1.0f - eps, t * (1.0f - eps));
            }
            else
            {
                uvs.emplace_back(t, t - eps);
            }
        }
        return uvs;
    }

    static glm::vec3 evaluateSurfaceRaw(
        const NagataPatch::NagataPatchData& patch,
        const NagataPatch::PatchEnhancementData& enhance,
        float u, float v)
    {
        NagataPatch::BlendingResult blend;
        NagataPatch::computeEffectiveCoefficients(patch, enhance, u, v, blend);

        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];

        float oneMinusU = 1.0f - u;
        float uMinusV = u - v;

        glm::vec3 P_lin = x00 * oneMinusU + x10 * uMinusV + x11 * v;
        glm::vec3 Q1 = blend.c_eff[0] * (oneMinusU * uMinusV);
        glm::vec3 Q2 = blend.c_eff[1] * (uMinusV * v);
        glm::vec3 Q3 = blend.c_eff[2] * (oneMinusU * v);

        return P_lin - Q1 - Q2 - Q3;
    }

    static bool checkEdgeConstraintsForTriangle(
        const glm::vec3& x00,
        const glm::vec3& x10,
        const glm::vec3& x11,
        const glm::vec3& c1_orig,
        const glm::vec3& c2_orig,
        const glm::vec3& c3_orig,
        int edgeIdx,
        const glm::vec3& c_sharp_edge,
        float k_factor,
        float eps = 1e-10f)
    {
        glm::vec3 n_ref = glm::cross(x10 - x00, x11 - x00);
        float n_len = glm::length(n_ref);
        if (n_len < 1e-12f)
        {
            return true;
        }
        n_ref /= n_len;

        NagataPatch::NagataPatchData patch;
        patch.vertices[0] = x00;
        patch.vertices[1] = x10;
        patch.vertices[2] = x11;
        patch.normals[0] = glm::vec3(0.0f);
        patch.normals[1] = glm::vec3(0.0f);
        patch.normals[2] = glm::vec3(0.0f);
        patch.c_orig[0] = c1_orig;
        patch.c_orig[1] = c2_orig;
        patch.c_orig[2] = c3_orig;

        NagataPatch::PatchEnhancementData enhance;
        enhance.edges[0].enabled = false;
        enhance.edges[1].enabled = false;
        enhance.edges[2].enabled = false;
        enhance.edges[edgeIdx].enabled = true;
        enhance.edges[edgeIdx].c_sharp = c_sharp_edge;
        enhance.edges[edgeIdx].k_factor = k_factor;

        glm::vec3 a, b, opp;
        if (edgeIdx == 0)
        {
            a = x00; b = x10; opp = x11;
        }
        else if (edgeIdx == 1)
        {
            a = x10; b = x11; opp = x00;
        }
        else
        {
            a = x00; b = x11; opp = x10;
        }

        glm::vec3 e = b - a;
        glm::vec3 sideDir = glm::cross(n_ref, e);
        float sideLen = glm::length(sideDir);
        if (sideLen < 1e-12f)
        {
            return true;
        }
        sideDir /= sideLen;
        if (glm::dot(sideDir, opp - (a + b) * 0.5f) < 0.0f)
        {
            sideDir = -sideDir;
        }

        const auto samples = sampleUvForEdge(edgeIdx);
        for (const auto& uv : samples)
        {
            float u = uv.first;
            float v = uv.second;

            glm::vec3 p = evaluateSurfaceRaw(patch, enhance, u, v);
            glm::vec3 dXdu, dXdv;
            NagataPatch::evaluateDerivatives(patch, enhance, u, v, dXdu, dXdv);
            float jac = glm::dot(glm::cross(dXdu, dXdv), n_ref);
            if (jac <= 0.0f)
            {
                return false;
            }

            float t = (edgeIdx == 0) ? u : v;
            glm::vec3 edgePoint = (1.0f - t) * a + t * b - c_sharp_edge * t * (1.0f - t);
            float s = glm::dot(p - edgePoint, sideDir);
            if (s < -eps)
            {
                return false;
            }
        }

        return true;
    }

    EnhancedNagataData computeCSharpForEdges(
        const std::map<EdgeKey, CreaseEdgeInfo>& creaseEdges,
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        float k_factor)
    {
        EnhancedNagataData data;

        for (const auto& [edgeKey, info] : creaseEdges)
        {
            glm::vec3 e = info.B - info.A;

            glm::vec3 dA = computeCreaseDirection(info.n_A_L, info.n_A_R, e);
            glm::vec3 dB = computeCreaseDirection(info.n_B_L, info.n_B_R, e);
            if (glm::dot(dA, dB) < 0.0f)
            {
                dB = -dB;
            }

            glm::vec3 c_sharp = computeCSharp(info.A, info.B, dA, dB);

            int triL = info.tri_L;
            int triR = info.tri_R;
            if (triL < 0 || triR < 0 ||
                triL >= static_cast<int>(faces.size()) ||
                triR >= static_cast<int>(faces.size()))
            {
                data.c_sharps[edgeKey] = c_sharp;
                continue;
            }

            const auto& triLIdx = faces[triL];
            const auto& triRIdx = faces[triR];
            int edgeIdxL = edgeIndexForTriangle(triLIdx, edgeKey);
            int edgeIdxR = edgeIndexForTriangle(triRIdx, edgeKey);
            if (edgeIdxL < 0 || edgeIdxR < 0)
            {
                data.c_sharps[edgeKey] = c_sharp;
                continue;
            }

            NagataPatch::NagataPatchData patchL(
                vertices[triLIdx[0]], vertices[triLIdx[1]], vertices[triLIdx[2]],
                faceNormals[triL][0], faceNormals[triL][1], faceNormals[triL][2]);
            NagataPatch::NagataPatchData patchR(
                vertices[triRIdx[0]], vertices[triRIdx[1]], vertices[triRIdx[2]],
                faceNormals[triR][0], faceNormals[triR][1], faceNormals[triR][2]);

            glm::vec3 c_orig_L = patchL.c_orig[edgeIdxL];
            glm::vec3 c_orig_R = patchR.c_orig[edgeIdxR];
            glm::vec3 baseline = 0.5f * (c_orig_L + c_orig_R);

            float low = 0.0f;
            float high = 1.0f;
            glm::vec3 best = baseline;
            for (int i = 0; i < 12; ++i)
            {
                float mid = 0.5f * (low + high);
                glm::vec3 candidate = baseline + mid * (c_sharp - baseline);

                bool okL = checkEdgeConstraintsForTriangle(
                    patchL.vertices[0], patchL.vertices[1], patchL.vertices[2],
                    patchL.c_orig[0], patchL.c_orig[1], patchL.c_orig[2],
                    edgeIdxL,
                    candidate,
                    k_factor);
                bool okR = checkEdgeConstraintsForTriangle(
                    patchR.vertices[0], patchR.vertices[1], patchR.vertices[2],
                    patchR.c_orig[0], patchR.c_orig[1], patchR.c_orig[2],
                    edgeIdxR,
                    candidate,
                    k_factor);

                if (okL && okR)
                {
                    best = candidate;
                    low = mid;
                }
                else
                {
                    high = mid;
                }
            }

            data.c_sharps[edgeKey] = best;
        }

        return data;
    }
    
    // ============================================================
    // Main Entry API
    // ============================================================
    
    EnhancedNagataData computeOrLoadEnhancedData(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        const std::string& nsmPath,
        bool saveCache)
    {
        std::string engPath = getEngFilepath(nsmPath);
        
        // Try to load cache
        EnhancedNagataData data;
        if (loadEnhancedData(engPath, data))
        {
            return data;
        }
        
        // Compute
        std::cout << "NagataEnhanced: Detecting crease edges..." << std::endl;
        auto creaseEdges = detectCreaseEdges(vertices, faces, faceNormals);
        std::cout << "NagataEnhanced: Found " << creaseEdges.size() << " crease edges" << std::endl;
        
        if (creaseEdges.empty())
        {
            return data;  // Empty data
        }
        
        std::cout << "NagataEnhanced: Computing shared boundary coefficients..." << std::endl;
        data = computeCSharpForEdges(creaseEdges, vertices, faces, faceNormals);
        
        if (saveCache)
        {
            saveEnhancedData(engPath, data);
        }
        
        return data;
    }
}
}
