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
        
        if (version != ENG_VERSION && version != ENG_VERSION_DOUBLE)
        {
            std::cerr << "NagataEnhanced: Unsupported ENG version " << version << std::endl;
            return false;
        }
        
        data.c_sharps.clear();
        for (uint32_t i = 0; i < numEdges; ++i)
        {
            uint32_t v0, v1;
            file.read(reinterpret_cast<char*>(&v0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&v1), sizeof(uint32_t));
            
            if (version == ENG_VERSION_DOUBLE)
            {
                double cx, cy, cz;
                file.read(reinterpret_cast<char*>(&cx), sizeof(double));
                file.read(reinterpret_cast<char*>(&cy), sizeof(double));
                file.read(reinterpret_cast<char*>(&cz), sizeof(double));
                
                if (!file)
                {
                    std::cerr << "NagataEnhanced: ENG file data incomplete" << std::endl;
                    return false;
                }
                
                data.c_sharps[EdgeKey(v0, v1)] = glm::vec3(
                    static_cast<float>(cx),
                    static_cast<float>(cy),
                    static_cast<float>(cz));
            }
            else
            {
                float cx, cy, cz;
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
        }
        
        std::cout << "NagataEnhanced: Loaded " << numEdges << " crease edges from " << filepath << std::endl;
        return true;
    }

    bool loadEnhancedDataDouble(const std::string& filepath, std::map<EdgeKey, glm::dvec3>& data)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }
        
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
        
        if (version != ENG_VERSION && version != ENG_VERSION_DOUBLE)
        {
            std::cerr << "NagataEnhanced: Unsupported ENG version " << version << std::endl;
            return false;
        }
        
        data.clear();
        for (uint32_t i = 0; i < numEdges; ++i)
        {
            uint32_t v0, v1;
            file.read(reinterpret_cast<char*>(&v0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&v1), sizeof(uint32_t));
            
            if (version == ENG_VERSION_DOUBLE)
            {
                double cx, cy, cz;
                file.read(reinterpret_cast<char*>(&cx), sizeof(double));
                file.read(reinterpret_cast<char*>(&cy), sizeof(double));
                file.read(reinterpret_cast<char*>(&cz), sizeof(double));
                
                if (!file)
                {
                    std::cerr << "NagataEnhanced: ENG file data incomplete" << std::endl;
                    return false;
                }
                
                data[EdgeKey(v0, v1)] = glm::dvec3(cx, cy, cz);
            }
            else
            {
                float cx, cy, cz;
                file.read(reinterpret_cast<char*>(&cx), sizeof(float));
                file.read(reinterpret_cast<char*>(&cy), sizeof(float));
                file.read(reinterpret_cast<char*>(&cz), sizeof(float));
                
                if (!file)
                {
                    std::cerr << "NagataEnhanced: ENG file data incomplete" << std::endl;
                    return false;
                }
                
                data[EdgeKey(v0, v1)] = glm::dvec3(
                    static_cast<double>(cx),
                    static_cast<double>(cy),
                    static_cast<double>(cz));
            }
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

    bool saveEnhancedDataDouble(const std::string& filepath, const std::map<EdgeKey, glm::dvec3>& data)
    {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "NagataEnhanced: Cannot create file " << filepath << std::endl;
            return false;
        }
        
        file.write(ENG_MAGIC, 4);
        uint32_t version = ENG_VERSION_DOUBLE;
        uint32_t numEdges = static_cast<uint32_t>(data.size());
        uint32_t reserved = 0;
        
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&numEdges), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&reserved), sizeof(uint32_t));
        
        for (const auto& [key, c] : data)
        {
            uint32_t v0 = key.v0;
            uint32_t v1 = key.v1;
            double cx = c.x, cy = c.y, cz = c.z;
            
            file.write(reinterpret_cast<const char*>(&v0), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&v1), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&cx), sizeof(double));
            file.write(reinterpret_cast<const char*>(&cy), sizeof(double));
            file.write(reinterpret_cast<const char*>(&cz), sizeof(double));
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
            
            auto getNormalAtVertex = [&](int triIdx, uint32_t globalVIdx) -> glm::vec3
            {
                const auto& tri = faces[triIdx];
                for (int i = 0; i < 3; ++i)
                {
                    if (tri[i] == globalVIdx)
                        return faceNormals[triIdx][i];
                }
                return glm::vec3(0.0f, 0.0f, 1.0f);
            };
            
            glm::vec3 n_A_L = getNormalAtVertex(triL.triIdx, edgeKey.v0);
            glm::vec3 n_B_L = getNormalAtVertex(triL.triIdx, edgeKey.v1);
            glm::vec3 n_A_R = getNormalAtVertex(triR.triIdx, edgeKey.v0);
            glm::vec3 n_B_R = getNormalAtVertex(triR.triIdx, edgeKey.v1);
            
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

    std::map<EdgeKey, CreaseEdgeInfoD> detectCreaseEdgesD(
        const std::vector<glm::dvec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::dvec3, 3>>& faceNormals,
        double gapThreshold)
    {
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
        
        std::map<EdgeKey, CreaseEdgeInfoD> creaseEdges;
        
        for (const auto& [edgeKey, trisInfo] : edgeToTris)
        {
            if (trisInfo.size() != 2)
                continue;
            
            const auto& triL = trisInfo[0];
            const auto& triR = trisInfo[1];
            
            glm::dvec3 A = vertices[edgeKey.v0];
            glm::dvec3 B = vertices[edgeKey.v1];
            
            auto getNormalAtVertex = [&](int triIdx, uint32_t globalVIdx) -> glm::dvec3
            {
                const auto& tri = faces[triIdx];
                for (int i = 0; i < 3; ++i)
                {
                    if (tri[i] == globalVIdx)
                        return faceNormals[triIdx][i];
                }
                return glm::dvec3(0.0, 0.0, 1.0);
            };
            
            glm::dvec3 n_A_L = getNormalAtVertex(triL.triIdx, edgeKey.v0);
            glm::dvec3 n_B_L = getNormalAtVertex(triL.triIdx, edgeKey.v1);
            glm::dvec3 n_A_R = getNormalAtVertex(triR.triIdx, edgeKey.v0);
            glm::dvec3 n_B_R = getNormalAtVertex(triR.triIdx, edgeKey.v1);
            
            glm::dvec3 e = B - A;
            glm::dvec3 c_L = NagataPatch::computeCurvatureD(e, n_A_L, n_B_L);
            glm::dvec3 c_R = NagataPatch::computeCurvatureD(e, n_A_R, n_B_R);
            
            double maxGap = 0.0;
            for (int i = 0; i <= 10; ++i)
            {
                double t = static_cast<double>(i) / 10.0;
                glm::dvec3 p_L = (1.0 - t) * A + t * B - c_L * t * (1.0 - t);
                glm::dvec3 p_R = (1.0 - t) * A + t * B - c_R * t * (1.0 - t);
                double gap = glm::length(p_L - p_R);
                maxGap = std::max(maxGap, gap);
            }
            
            if (maxGap > gapThreshold)
            {
                CreaseEdgeInfoD info;
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

    glm::dvec3 computeCreaseDirectionD(glm::dvec3 nL, glm::dvec3 nR, glm::dvec3 e)
    {
        glm::dvec3 d = glm::cross(nL, nR);
        double len = glm::length(d);
        
        if (len < 1e-8)
        {
            return glm::normalize(e);
        }
        
        d = d / len;
        
        if (glm::dot(d, e) < 0.0)
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
        if (std::abs(det) < 1e-12f)
        {
            return glm::vec3(0.0f);
        }
        
        float lA = (G11 * r0 - G01 * r1) / det;
        float lB = (-G01 * r0 + G00 * r1) / det;
        
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

    glm::dvec3 computeCSharpD(glm::dvec3 A, glm::dvec3 B, glm::dvec3 dA, glm::dvec3 dB)
    {
        glm::dvec3 e = B - A;
        
        if (glm::dot(dA, dB) < 0.0)
            dB = -dB;
        
        double G00 = glm::dot(dA, dA);
        double G01 = glm::dot(dA, dB);
        double G11 = glm::dot(dB, dB);
        double r0 = 2.0 * glm::dot(e, dA);
        double r1 = 2.0 * glm::dot(e, dB);
        
        double lambda = 1e-6;
        G00 += lambda;
        G11 += lambda;
        
        double det = G00 * G11 - G01 * G01;
        if (std::abs(det) < 1e-12)
        {
            return glm::dvec3(0.0);
        }
        
        double lA = (G11 * r0 - G01 * r1) / det;
        double lB = (-G01 * r0 + G00 * r1) / det;
        
        glm::dvec3 T_A = lA * dA;
        glm::dvec3 T_B = lB * dB;
        
        glm::dvec3 c_sharp = 0.5 * (T_B - T_A);
        
        double eLen = glm::length(e);
        double cLen = glm::length(c_sharp);
        double maxC = 2.0 * eLen;
        if (cLen > maxC)
        {
            c_sharp = c_sharp * (maxC / cLen);
        }
        
        return c_sharp;
    }
    
    EnhancedNagataData computeCSharpForEdges(
        const std::map<EdgeKey, CreaseEdgeInfo>& creaseEdges)
    {
        EnhancedNagataData data;
        
        for (const auto& [edgeKey, info] : creaseEdges)
        {
            glm::vec3 e = info.B - info.A;
            
            glm::vec3 dA = computeCreaseDirection(info.n_A_L, info.n_A_R, e);
            glm::vec3 dB = computeCreaseDirection(info.n_B_L, info.n_B_R, e);
            
            glm::vec3 c_sharp = computeCSharp(info.A, info.B, dA, dB);
            
            data.c_sharps[edgeKey] = c_sharp;
        }
        
        return data;
    }

    std::map<EdgeKey, glm::dvec3> computeCSharpForEdgesD(
        const std::map<EdgeKey, CreaseEdgeInfoD>& creaseEdges)
    {
        std::map<EdgeKey, glm::dvec3> data;
        
        for (const auto& [edgeKey, info] : creaseEdges)
        {
            glm::dvec3 e = info.B - info.A;
            
            glm::dvec3 dA = computeCreaseDirectionD(info.n_A_L, info.n_A_R, e);
            glm::dvec3 dB = computeCreaseDirectionD(info.n_B_L, info.n_B_R, e);
            
            glm::dvec3 c_sharp = computeCSharpD(info.A, info.B, dA, dB);
            
            data[edgeKey] = c_sharp;
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
        data = computeCSharpForEdges(creaseEdges);
        
        if (saveCache)
        {
            saveEnhancedData(engPath, data);
        }
        
        return data;
    }
    
    // ============================================================
    // Enhanced surface evaluation
    // ============================================================
    
    float smoothstep(float t)
    {
        t = std::clamp(t, 0.0f, 1.0f);
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    float gaussianDecay(float d, float k)
    {
        return std::exp(-k * d * d);
    }

    float gaussianDecayDeriv(float d, float k)
    {
        float w = gaussianDecay(d, k);
        return -2.0f * k * d * w;
    }
    
    glm::vec3 evaluateSurfaceEnhanced(
        const NagataPatch::NagataPatchData& patch,
        float u, float v,
        glm::vec3 c_sharp_1, glm::vec3 c_sharp_2, glm::vec3 c_sharp_3,
        std::array<bool, 3> isCrease,
        float k_factor)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];
        
        glm::vec3 c1_orig = patch.c_orig[0]; // Correction: accessing c_orig directly, not curvatureCoeffs
        glm::vec3 c2_orig = patch.c_orig[1];
        glm::vec3 c3_orig = patch.c_orig[2];
        
        // Distance parameters
        float d1 = v;           // Edge 1
        float d2 = 1.0f - u;    // Edge 2
        float d3 = u - v;       // Edge 3
        
        auto blendCoeff = [&](glm::vec3 c_orig, glm::vec3 c_sharp, float d, bool isCreaseEdge) -> glm::vec3
        {
            if (!isCreaseEdge)
                return c_orig;
            
            float w = gaussianDecay(d, k_factor);
            return c_orig + (c_sharp - c_orig) * w;
        };
        
        glm::vec3 c1_eff = blendCoeff(c1_orig, c_sharp_1, d1, isCrease[0]);
        glm::vec3 c2_eff = blendCoeff(c2_orig, c_sharp_2, d2, isCrease[1]);
        glm::vec3 c3_eff = blendCoeff(c3_orig, c_sharp_3, d3, isCrease[2]);
        
        float oneMinusU = 1.0f - u;
        float uMinusV = u - v;
        
        return x00 * oneMinusU + x10 * uMinusV + x11 * v
             - c1_eff * oneMinusU * uMinusV
             - c2_eff * uMinusV * v
             - c3_eff * oneMinusU * v;
    }

    float smoothstepDeriv(float t)
    {
        if (t <= 0.0f || t >= 1.0f) return 0.0f;
        float v = t * (1.0f - t);
        return 30.0f * v * v;
    }

    void evaluateDerivativesEnhanced(
        const NagataPatch::NagataPatchData& patch,
        float u, float v,
        glm::vec3 c_sharp_1, glm::vec3 c_sharp_2, glm::vec3 c_sharp_3,
        std::array<bool, 3> isCrease,
        float k_factor,
        glm::vec3& dXdu, glm::vec3& dXdv)
    {
        const glm::vec3& x00 = patch.vertices[0];
        const glm::vec3& x10 = patch.vertices[1];
        const glm::vec3& x11 = patch.vertices[2];

        float d1 = v;           // Edge 1
        float d2 = 1.0f - u;    // Edge 2
        float d3 = u - v;       // Edge 3

        auto getCoeff = [&](const glm::vec3& c_orig, const glm::vec3& c_sharp, float d, bool isCrease) 
        {
            if (!isCrease) return c_orig;
            float w = gaussianDecay(d, k_factor);
            return c_orig + (c_sharp - c_orig) * w;
        };

        glm::vec3 c1 = getCoeff(patch.c_orig[0], c_sharp_1, d1, isCrease[0]);
        glm::vec3 c2 = getCoeff(patch.c_orig[1], c_sharp_2, d2, isCrease[1]);
        glm::vec3 c3 = getCoeff(patch.c_orig[2], c_sharp_3, d3, isCrease[2]);

        glm::vec3 dXdu_base = -x00 + x10 
                            - c1 * (1.0f - 2.0f * u + v) 
                            - c2 * v 
                            + c3 * v;

        glm::vec3 dXdv_base = -x10 + x11 
                            + c1 * (1.0f - u) 
                            - c2 * (u - 2.0f * v) 
                            - c3 * (1.0f - u);
        
        auto getDampingDeriv = [&](float d, bool isCrease) -> float
        {
            if (!isCrease) return 0.0f;
            return gaussianDecayDeriv(d, k_factor);
        };

        glm::vec3 delta1 = c_sharp_1 - patch.c_orig[0];
        glm::vec3 delta2 = c_sharp_2 - patch.c_orig[1];
        glm::vec3 delta3 = c_sharp_3 - patch.c_orig[2];

        float dd1_dv = getDampingDeriv(d1, isCrease[0]);
        glm::vec3 dc1_dv = delta1 * dd1_dv;

        float dd2_du = getDampingDeriv(d2, isCrease[1]) * (-1.0f);
        glm::vec3 dc2_du = delta2 * dd2_du;

        float dd3_raw = getDampingDeriv(d3, isCrease[2]);
        glm::vec3 dc3_du = delta3 * dd3_raw;
        glm::vec3 dc3_dv = delta3 * (-dd3_raw);

        float B1 = (1.0f - u) * (u - v);
        float B2 = (u - v) * v;
        float B3 = (1.0f - u) * v;

        dXdu = dXdu_base - (dc2_du * B2 + dc3_du * B3);
        dXdv = dXdv_base - (dc1_dv * B1 + dc3_dv * B3);
    }

    glm::vec3 evaluateNormalEnhanced(
        const NagataPatch::NagataPatchData& patch,
        float u, float v,
        glm::vec3 c_sharp_1, glm::vec3 c_sharp_2, glm::vec3 c_sharp_3,
        std::array<bool, 3> isCrease,
        float k_factor)
    {
        glm::vec3 dXdu, dXdv;
        evaluateDerivativesEnhanced(patch, u, v, c_sharp_1, c_sharp_2, c_sharp_3, isCrease, k_factor, dXdu, dXdv);
        
        glm::vec3 normal = glm::cross(dXdu, dXdv);
        float len = glm::length(normal);
        
        if (len > 1e-10f) return normal / len;
        
        float w0 = 1.0f - u;
        float w1 = u - v;
        float w2 = v;
        return glm::normalize(w0 * patch.normals[0] + w1 * patch.normals[1] + w2 * patch.normals[2]);
    }

    float findNearestPointOnEnhancedNagataPatch(
        glm::vec3 point,
        const NagataPatch::NagataPatchData& patch,
        const EnhancedNagataData& enhancedData,
        const std::array<uint32_t, 3>& vertexIndices,
        glm::vec3& nearestPoint, 
        float& minU, float& minV)
    {
        EdgeKey key1(vertexIndices[0], vertexIndices[1]);
        EdgeKey key2(vertexIndices[1], vertexIndices[2]);
        EdgeKey key3(vertexIndices[0], vertexIndices[2]);
        
        std::array<bool, 3> isCrease = {
            enhancedData.hasEdge(key1),
            enhancedData.hasEdge(key2),
            enhancedData.hasEdge(key3)
        };

        if (!isCrease[0] && !isCrease[1] && !isCrease[2])
        {
             NagataPatch::PatchEnhancementData emptyEnhance; 
             auto res = NagataPatch::findNearestPoint(point, patch, emptyEnhance);
             nearestPoint = res.nearestPoint;
             minU = res.parameter.x;
             minV = res.parameter.y;
             return res.sqDistance;
        }

        glm::vec3 c1 = enhancedData.getCSharpOriented(vertexIndices[0], vertexIndices[1]);
        glm::vec3 c2 = enhancedData.getCSharpOriented(vertexIndices[1], vertexIndices[2]);
        glm::vec3 c3 = enhancedData.getCSharpOriented(vertexIndices[0], vertexIndices[2]);
        float k_factor = 0.0f; 

        // Initial guess: multi-point sampling
        minU = 0.333f; minV = 0.166f;
        float minDistSq = std::numeric_limits<float>::max();
        
        const std::array<std::pair<float, float>, 7> initialSamples = {{
            {0.5f, 0.25f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
            {0.5f, 0.0f}, {1.0f, 0.5f}, {0.5f, 0.5f}
        }};
        
        for (const auto& sample : initialSamples)
        {
            float u = sample.first;
            float v = sample.second;
            
            for (int iter = 0; iter < 10; ++iter)
            {
                glm::vec3 surfacePoint = evaluateSurfaceEnhanced(patch, u, v, c1, c2, c3, isCrease, k_factor);
                glm::vec3 diffVec = surfacePoint - point;

                glm::vec3 dXdu, dXdv;
                evaluateDerivativesEnhanced(patch, u, v, c1, c2, c3, isCrease, k_factor, dXdu, dXdv);
                
                float gradU = glm::dot(diffVec, dXdu);
                float gradV = glm::dot(diffVec, dXdv);
                
                float H11 = glm::dot(dXdu, dXdu);
                float H12 = glm::dot(dXdu, dXdv);
                float H22 = glm::dot(dXdv, dXdv);
                
                float det = H11 * H22 - H12 * H12;
                if (std::abs(det) < 1e-10f) break;
                
                float deltaU = (H22 * (-gradU) - H12 * (-gradV)) / det;
                float deltaV = (-H12 * (-gradU) + H11 * (-gradV)) / det;
                
                u += deltaU;
                v += deltaV;
                
                if (v < 0.0f) v = 0.0f;
                if (u < 0.0f) u = 0.0f;
                if (u > 1.0f) u = 1.0f;
                if (v > u) v = u;

                if (std::abs(deltaU) < 1e-5f && std::abs(deltaV) < 1e-5f) break;
            }
            
            glm::vec3 p = evaluateSurfaceEnhanced(patch, u, v, c1, c2, c3, isCrease, k_factor);
            float d2 = glm::dot(point - p, point - p);
            if (d2 < minDistSq)
            {
                minDistSq = d2;
                minU = u; minV = v;
                nearestPoint = p;
            }
        }
        
        return minDistSq;
    }

    float getSignedDistPointAndEnhancedNagataPatch(
        glm::vec3 point,
        const NagataPatch::NagataPatchData& patch,
        const EnhancedNagataData& enhancedData,
        const std::array<uint32_t, 3>& vertexIndices,
        glm::vec3* outNearestPoint)
    {
        glm::vec3 nearestPoint;
        float u, v;
        
        float distSq = findNearestPointOnEnhancedNagataPatch(point, patch, enhancedData, vertexIndices, nearestPoint, u, v);
        
        EdgeKey key1(vertexIndices[0], vertexIndices[1]);
        EdgeKey key2(vertexIndices[1], vertexIndices[2]);
        EdgeKey key3(vertexIndices[0], vertexIndices[2]);
        std::array<bool, 3> isCrease = { enhancedData.hasEdge(key1), enhancedData.hasEdge(key2), enhancedData.hasEdge(key3) };
        glm::vec3 c1 = enhancedData.getCSharpOriented(vertexIndices[0], vertexIndices[1]);
        glm::vec3 c2 = enhancedData.getCSharpOriented(vertexIndices[1], vertexIndices[2]);
        glm::vec3 c3 = enhancedData.getCSharpOriented(vertexIndices[0], vertexIndices[2]);

        glm::vec3 normal = evaluateNormalEnhanced(patch, u, v, c1, c2, c3, isCrease);
        
        float sign = glm::dot(point - nearestPoint, normal) >= 0.0f ? 1.0f : -1.0f;
        
        if (outNearestPoint) *outNearestPoint = nearestPoint;
        
        return sign * std::sqrt(distSq);
    }
}
}
