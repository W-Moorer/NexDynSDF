#ifndef NAGATA_ENHANCED_H
#define NAGATA_ENHANCED_H

/**
 * @file NagataEnhanced.h
 * @brief Nagata Enhanced Module - Crease edge handling and ENG file cache
 * 
 * Features:
 * - .eng file I/O (Enhanced Nagata Geometry)
 * - Crease edge detection
 * - Shared boundary coefficients calculation (c_sharp)
 * - Automatic caching logic
 */

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <string>
#include "sdflib/utils/NagataPatch.h"

namespace sdflib
{
namespace NagataEnhanced
{
    // ============================================================
    // Constant definitions
    // ============================================================
    
    constexpr char ENG_MAGIC[4] = {'E', 'N', 'G', '\0'};
    constexpr uint32_t ENG_VERSION = 1;
    constexpr float GAP_THRESHOLD = 1e-4f;
    
    // ============================================================
    // Data structures
    // ============================================================
    
    /**
     * @brief Edge key (sorted vertex index pair)
     */
    struct EdgeKey
    {
        uint32_t v0;
        uint32_t v1;
        
        EdgeKey() : v0(0), v1(0) {}
        EdgeKey(uint32_t a, uint32_t b) : v0(std::min(a,b)), v1(std::max(a,b)) {}
        
        bool operator<(const EdgeKey& other) const
        {
            if (v0 != other.v0) return v0 < other.v0;
            return v1 < other.v1;
        }
        
        bool operator==(const EdgeKey& other) const
        {
            return v0 == other.v0 && v1 == other.v1;
        }
    };
    
    /**
     * @brief Crease edge info
     */
    struct CreaseEdgeInfo
    {
        glm::vec3 A, B;           // Endpoint coordinates
        glm::vec3 n_A_L, n_A_R;   // Normals at A (Left/Right)
        glm::vec3 n_B_L, n_B_R;   // Normals at B (Left/Right)
        int tri_L, tri_R;         // Adjacent triangle indices
        float max_gap;            // Maximum gap
    };
    
    /**
     * @brief Enhanced Nagata data
     */
    struct EnhancedNagataData
    {
        std::map<EdgeKey, glm::vec3> c_sharps;  // Edge -> c_sharp coefficients
        
        bool hasEdge(const EdgeKey& key) const
        {
            return c_sharps.find(key) != c_sharps.end();
        }
        
        glm::vec3 getCSharp(const EdgeKey& key) const
        {
            auto it = c_sharps.find(key);
            if (it != c_sharps.end()) return it->second;
            return glm::vec3(0.0f);
        }

        glm::vec3 getCSharpOriented(uint32_t a, uint32_t b) const
        {
            EdgeKey key(a, b);
            glm::vec3 c = getCSharp(key);
            return (a <= b) ? c : -c;
        }
        
        size_t size() const { return c_sharps.size(); }
        bool empty() const { return c_sharps.empty(); }
    };
    
    // ============================================================
    // ENG file I/O
    // ============================================================
    
    /**
     * @brief Load enhanced data from .eng file
     */
    bool loadEnhancedData(const std::string& filepath, EnhancedNagataData& data);
    
    /**
     * @brief Save enhanced data to .eng file
     */
    bool saveEnhancedData(const std::string& filepath, const EnhancedNagataData& data);
    
    /**
     * @brief Derive ENG filepath from NSM path
     */
    std::string getEngFilepath(const std::string& nsmPath);
    
    /**
     * @brief Check if ENG cache file exists
     */
    bool hasEngCache(const std::string& nsmPath);
    
    // ============================================================
    // Crease edge detection
    // ============================================================
    
    /**
     * @brief Detect crease edges
     * 
     * @param vertices Vertex coordinates
     * @param faces Face indices
     * @param faceNormals Face vertex normals
     * @param gapThreshold Gap threshold
     */
    std::map<EdgeKey, CreaseEdgeInfo> detectCreaseEdges(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        float gapThreshold = GAP_THRESHOLD);
    
    // ============================================================
    // c_sharp calculation
    // ============================================================
    
    /**
     * @brief Compute crease direction (intersection of two tangent planes)
     */
    glm::vec3 computeCreaseDirection(glm::vec3 nL, glm::vec3 nR, glm::vec3 e);
    
    /**
     * @brief Compute shared boundary coefficient c_sharp
     * 
     * Solve for endpoint tangent lengths using least squares
     */
    glm::vec3 computeCSharp(glm::vec3 A, glm::vec3 B, glm::vec3 dA, glm::vec3 dB);
    
    /**
     * @brief Compute c_sharp for all crease edges
     */
    EnhancedNagataData computeCSharpForEdges(
        const std::map<EdgeKey, CreaseEdgeInfo>& creaseEdges,
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        float k_factor = 0.1f);
    
    // ============================================================
    // Main Entry API
    // ============================================================
    
    /**
     * @brief Compute or load enhanced data
     * 
     * - Load cache if .eng exists
     * - Otherwise compute and save cache
     */
    EnhancedNagataData computeOrLoadEnhancedData(
        const std::vector<glm::vec3>& vertices,
        const std::vector<std::array<uint32_t, 3>>& faces,
        const std::vector<std::array<glm::vec3, 3>>& faceNormals,
        const std::string& nsmPath,
        bool saveCache = true);
    
}
}

#endif // NAGATA_ENHANCED_H
