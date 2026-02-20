/**
 * @file test_hybrid_accuracy.cpp
 * @brief Accuracy test comparing OctreeSDF (flat) vs Hybrid (Nagata) on sphere
 * 
 * Uses sphere.nsm (radius=1.0, center=origin) to test precision improvement
 */

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <array>
#include <cmath>
#include <glm/glm.hpp>

#include "sdflib/utils/MeshBinaryLoader.h"
#include "sdflib/utils/Mesh.h"
#include "sdflib/OctreeSdf.h"
#include "sdflib/HybridOctreeSdf.h"
#include "sdflib/utils/NagataPatch.h"
#include "sdflib/utils/NagataEnhanced.h"

using namespace sdflib;

// Helper: Generate random points in narrowband
std::vector<glm::vec3> generateNarrowbandPoints(float radius, float bandwidth, int count, unsigned seed = 42)
{
    std::vector<glm::vec3> points;
    points.reserve(count);
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distPhi(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> distTheta(0.0f, 3.14159265f);
    std::uniform_real_distribution<float> distOffset(-bandwidth, bandwidth);
    
    for (int i = 0; i < count; ++i)
    {
        // Random point on sphere surface
        float theta = distTheta(rng);
        float phi = distPhi(rng);
        
        float x = std::sin(theta) * std::cos(phi);
        float y = std::sin(theta) * std::sin(phi);
        float z = std::cos(theta);
        
        glm::vec3 surfacePoint(x, y, z);
        surfacePoint *= radius;
        
        // Add random offset in normal direction
        float offset = distOffset(rng);
        glm::vec3 testPoint = surfacePoint + glm::normalize(surfacePoint) * offset;
        
        points.push_back(testPoint);
    }
    
    return points;
}

static bool test_crease_csharp_orientation()
{
    const std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
    };

    const std::vector<std::array<uint32_t, 3>> faces = {
        {0, 1, 2},
        {0, 2, 3},
    };

    const glm::vec3 n0(0.0f, 0.0f, 1.0f);
    const glm::vec3 nA1 = glm::normalize(glm::vec3(0.3f, 0.0f, 1.0f));
    const glm::vec3 nC0 = glm::normalize(glm::vec3(0.0f, 0.3f, 1.0f));
    const std::vector<std::array<glm::vec3, 3>> faceNormals = {
        {n0, n0, nC0},
        {nA1, n0, n0},
    };

    const float gapThreshold = 1.0e-10f;
    const auto creaseEdges = NagataEnhanced::detectCreaseEdges(vertices, faces, faceNormals, gapThreshold);
    if (creaseEdges.empty())
    {
        std::cerr << "Crease test: no crease edges detected" << std::endl;
        return false;
    }

    const auto enhancedData = NagataEnhanced::computeCSharpForEdges(creaseEdges, vertices, faces, faceNormals);
    if (enhancedData.empty())
    {
        std::cerr << "Crease test: c_sharp map is empty" << std::endl;
        return false;
    }

    const NagataPatch::NagataPatchData patch0(
        vertices[faces[0][0]], vertices[faces[0][1]], vertices[faces[0][2]],
        faceNormals[0][0], faceNormals[0][1], faceNormals[0][2]);

    const NagataPatch::NagataPatchData patch1(
        vertices[faces[1][0]], vertices[faces[1][1]], vertices[faces[1][2]],
        faceNormals[1][0], faceNormals[1][1], faceNormals[1][2]);

    const std::array<uint32_t, 3> vi0 = faces[0];
    const std::array<uint32_t, 3> vi1 = faces[1];

    auto eval_edge_point = [&](const NagataPatch::NagataPatchData& patch, const std::array<uint32_t, 3>& vi, float u, float v)
    {
        const NagataEnhanced::EdgeKey k01(vi[0], vi[1]);
        const NagataEnhanced::EdgeKey k12(vi[1], vi[2]);
        const NagataEnhanced::EdgeKey k02(vi[0], vi[2]);
        const float kFactor = 0.1f;

        const std::array<bool, 3> isCrease = {
            enhancedData.hasEdge(k01),
            enhancedData.hasEdge(k12),
            enhancedData.hasEdge(k02),
        };

        const glm::vec3 c1 = enhancedData.getCSharp(k01);
        const glm::vec3 c2 = enhancedData.getCSharp(k12);
        const glm::vec3 c3 = enhancedData.getCSharp(k02);

        NagataPatch::PatchEnhancementData enhance;
        enhance.edges[0].enabled = isCrease[0];
        enhance.edges[1].enabled = isCrease[1];
        enhance.edges[2].enabled = isCrease[2];
        enhance.edges[0].c_sharp = c1;
        enhance.edges[1].c_sharp = c2;
        enhance.edges[2].c_sharp = c3;
        enhance.edges[0].k_factor = kFactor;
        enhance.edges[1].k_factor = kFactor;
        enhance.edges[2].k_factor = kFactor;

        return NagataPatch::evaluateSurface(patch, enhance, u, v);
    };

    const float t = 0.3f;
    const glm::vec3 p0 = eval_edge_point(patch0, vi0, t, t);
    const glm::vec3 p1 = eval_edge_point(patch1, vi1, t, 0.0f);

    const float err = glm::length(p0 - p1);
    if (!(err < 1.0e-6f))
    {
        std::cerr << "Crease test: edge mismatch err=" << err
                  << " p0=(" << p0.x << "," << p0.y << "," << p0.z << ")"
                  << " p1=(" << p1.x << "," << p1.y << "," << p1.z << ")"
                  << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "Hybrid SDF Accuracy Test (Sphere)" << std::endl;
    std::cout << "========================================" << std::endl;

    if (!test_crease_csharp_orientation())
    {
        std::cerr << "Error: crease c_sharp orientation test failed" << std::endl;
        return 1;
    }
    
    // Load sphere model
    std::string nsmPath = "output/sphere.nsm";
    if (argc > 1)
    {
        nsmPath = argv[1];
    }

    auto resolveRelativePath = [&](const std::string& rel) -> std::filesystem::path
    {
        std::filesystem::path p(rel);
        if (p.is_absolute() && std::filesystem::exists(p)) return p;
        if (std::filesystem::exists(p)) return p;

        std::filesystem::path cur = std::filesystem::current_path();
        for (int i = 0; i < 8; ++i)
        {
            std::filesystem::path cand = cur / p;
            if (std::filesystem::exists(cand)) return cand;
            if (!cur.has_parent_path()) break;
            cur = cur.parent_path();
        }

        return p;
    };

    std::filesystem::path nsmResolved = resolveRelativePath(nsmPath);
    if (!std::filesystem::exists(nsmResolved))
    {
        std::cerr << "Error: NSM file not found: " << nsmPath << std::endl;
        return 1;
    }
    
    std::cout << "Loading sphere mesh: " << nsmResolved.string() << std::endl;
    MeshBinaryLoader::MeshData meshData = MeshBinaryLoader::loadFromNSM(nsmResolved.string());
    
    if (meshData.vertices.empty())
    {
        std::cerr << "Error: Failed to load mesh data." << std::endl;
        return 1;
    }
    
    // Convert to Mesh
    std::vector<uint32_t> indices;
    indices.reserve(meshData.faces.size() * 3);
    for (const auto& face : meshData.faces)
    {
        indices.push_back(face[0]);
        indices.push_back(face[1]);
        indices.push_back(face[2]);
    }
    
    Mesh mesh(meshData.vertices.data(), meshData.vertices.size(),
              indices.data(), indices.size());
    
    // Compute vertex normals
    std::vector<glm::vec3> meshNormals(meshData.vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < meshData.faces.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            uint32_t vIdx = meshData.faces[i][j];
            meshNormals[vIdx] += meshData.faceNormals[i][j];
        }
    }
    for (auto& n : meshNormals)
    {
        if (glm::length(n) > 1e-6f) n = glm::normalize(n);
    }
    mesh.getNormals() = meshNormals;
    mesh.computeBoundingBox();
    
    // Generate Nagata patches
    std::cout << "Generating Nagata patches..." << std::endl;
    std::vector<NagataPatch::NagataPatchData> patches;
    std::vector<NagataPatch::PatchEnhancementData> enhanced;
    
    patches.reserve(meshData.faces.size());
    enhanced.reserve(meshData.faces.size());
    
    for (size_t i = 0; i < meshData.faces.size(); ++i)
    {
        const auto& face = meshData.faces[i];
        patches.emplace_back(
            meshData.vertices[face[0]],
            meshData.vertices[face[1]],
            meshData.vertices[face[2]],
            meshData.faceNormals[i][0],
            meshData.faceNormals[i][1],
            meshData.faceNormals[i][2]);
        
        enhanced.emplace_back();  // No crease enhancement for smooth sphere
    }
    
    std::cout << "Generated " << patches.size() << " Nagata patches." << std::endl;
    
    // Setup bounding box
    BoundingBox box = mesh.getBoundingBox();
    glm::vec3 size = box.getSize();
    float maxDim = glm::max(glm::max(size.x, size.y), size.z);
    box.addMargin(maxDim * 0.1f);
    
    // Build Baseline OctreeSDF (flat triangles)
    std::cout << "\nBuilding Baseline OctreeSDF (flat triangles)..." << std::endl;
    OctreeSdf baseline(mesh, box, 6, 2,
                       OctreeSdf::TerminationRule::TRAPEZOIDAL_RULE,
                       OctreeSdf::TerminationRuleParams::setTrapezoidalRuleParams(1e-3f),
                       OctreeSdf::InitAlgorithm::CONTINUITY);
    std::cout << "Baseline built." << std::endl;
    
    // Build Hybrid OctreeSDF (Nagata patches)
    std::cout << "Building Hybrid OctreeSDF (Nagata patches)..." << std::endl;
    HybridOctreeSdf hybrid(mesh, patches, enhanced, box, 6, 2,
                           OctreeSdf::TerminationRule::TRAPEZOIDAL_RULE,
                           OctreeSdf::TerminationRuleParams::setTrapezoidalRuleParams(1e-3f));
    std::cout << "Hybrid built." << std::endl;
    
    // Generate test points in narrowband
    const float bandwidth = 0.05f;  // ±5cm around sphere
    const int numPoints = 500;
    std::cout << "\nGenerating " << numPoints << " test points in narrowband (±" << bandwidth << ")..." << std::endl;
    std::vector<glm::vec3> testPoints = generateNarrowbandPoints(1.0f, bandwidth, numPoints);
    
    // Evaluate both SDFs
    std::cout << "\nEvaluating accuracy..." << std::endl;
    float sumErrorBaseline = 0.0f;
    float maxErrorBaseline = 0.0f;
    float sumErrorHybrid = 0.0f;
    float maxErrorHybrid = 0.0f;
    
    for (const auto& p : testPoints)
    {
        float dAnalytical = glm::length(p) - 1.0f;  // Exact distance for unit sphere
        float dBaseline = baseline.getDistance(p);
        float dHybrid = hybrid.getDistance(p);
        
        float errorBaseline = std::abs(dBaseline - dAnalytical);
        float errorHybrid = std::abs(dHybrid - dAnalytical);
        
        sumErrorBaseline += errorBaseline;
        maxErrorBaseline = std::max(maxErrorBaseline, errorBaseline);
        sumErrorHybrid += errorHybrid;
        maxErrorHybrid = std::max(maxErrorHybrid, errorHybrid);
    }
    
    float avgErrorBaseline = sumErrorBaseline / testPoints.size();
    float avgErrorHybrid = sumErrorHybrid / testPoints.size();
    
    // Print results
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Baseline (Flat triangles):" << std::endl;
    std::cout << "  Average error: " << avgErrorBaseline << std::endl;
    std::cout << "  Max error:     " << maxErrorBaseline << std::endl;
    std::cout << "\nHybrid (Nagata patches):" << std::endl;
    std::cout << "  Average error: " << avgErrorHybrid << std::endl;
    std::cout << "  Max error:     " << maxErrorHybrid << std::endl;
    std::cout << "\nImprovement:" << std::endl;
    std::cout << "  Average: " << 100.0f * (1.0f - avgErrorHybrid / avgErrorBaseline) << "%" << std::endl;
    std::cout << "  Max:     " << 100.0f * (1.0f - maxErrorHybrid / maxErrorBaseline) << "%" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Verification: Hybrid should be significantly more accurate
    if (avgErrorHybrid < avgErrorBaseline * 0.1f)
    {
        std::cout << "\n PASS: Hybrid accuracy is >10x better" << std::endl;
        return 0;
    }
    else if (avgErrorHybrid < avgErrorBaseline * 0.5f)
    {
        std::cout << "\n PARTIAL: Hybrid is better but <10x improvement" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\n FAIL: Hybrid accuracy not significantly improved" << std::endl;
        return 1;
    }
}
