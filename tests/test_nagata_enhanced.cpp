/**
 * @file test_nagata_enhanced.cpp
 * @brief Nagata 增强模块测试
 * 
 * 测试用例:
 * 1. ENG 文件读写往返
 * 2. 裂隙边检测
 * 3. c_sharp 计算
 * 4. 增强曲面求值
 */

#include <iostream>
#include <cmath>
#include <filesystem>
#include <glm/glm.hpp>
#include "SdfLib/utils/NagataEnhanced.h"
#include "SdfLib/utils/MeshBinaryLoader.h"
#include "SdfLib/utils/NagataPatch.h"

using namespace sdflib;

// ============================================================
// 测试 1: ENG 文件读写往返
// ============================================================
void testEngFileIO()
{
    std::cout << "\n=== Test 1: ENG File I/O ===" << std::endl;
    
    // 创建测试数据
    NagataEnhanced::EnhancedNagataData original;
    original.c_sharps[NagataEnhanced::EdgeKey(0, 1)] = glm::vec3(0.1f, 0.2f, 0.3f);
    original.c_sharps[NagataEnhanced::EdgeKey(1, 2)] = glm::vec3(0.4f, 0.5f, 0.6f);
    original.c_sharps[NagataEnhanced::EdgeKey(0, 2)] = glm::vec3(0.7f, 0.8f, 0.9f);
    
    std::string testFile = "test_output.eng";
    
    // 保存
    bool saved = NagataEnhanced::saveEnhancedData(testFile, original);
    if (!saved)
    {
        std::cout << "FAIL: Could not save ENG file" << std::endl;
        return;
    }
    
    // 加载
    NagataEnhanced::EnhancedNagataData loaded;
    bool loadSuccess = NagataEnhanced::loadEnhancedData(testFile, loaded);
    if (!loadSuccess)
    {
        std::cout << "FAIL: Could not load ENG file" << std::endl;
        return;
    }
    
    // 比较
    bool match = true;
    for (const auto& [key, val] : original.c_sharps)
    {
        if (!loaded.hasEdge(key))
        {
            std::cout << "FAIL: Missing edge (" << key.v0 << ", " << key.v1 << ")" << std::endl;
            match = false;
            continue;
        }
        
        glm::vec3 loadedVal = loaded.getCSharp(key);
        float diff = glm::length(val - loadedVal);
        if (diff > 1e-5f)
        {
            std::cout << "FAIL: Value mismatch for edge (" << key.v0 << ", " << key.v1 << ")" << std::endl;
            match = false;
        }
    }
    
    if (match && loaded.size() == original.size())
    {
        std::cout << "PASS: ENG file roundtrip successful" << std::endl;
    }
    
    // 清理
    std::filesystem::remove(testFile);
    std::cout << "Cleaned up test file" << std::endl;
}

// ============================================================
// 测试 2: NSM 加载与裂隙边检测
// ============================================================
void testCreaseEdgeDetection(const std::string& nsmPath)
{
    std::cout << "\n=== Test 2: Crease Edge Detection ===" << std::endl;
    std::cout << "Loading: " << nsmPath << std::endl;
    
    // 加载 NSM 文件
    auto meshData = MeshBinaryLoader::loadFromNSM(nsmPath);
    
    if (meshData.vertices.empty())
    {
        std::cout << "FAIL: Could not load NSM file" << std::endl;
        return;
    }
    
    // 检测裂隙边
    auto creaseEdges = NagataEnhanced::detectCreaseEdges(
        meshData.vertices, meshData.faces, meshData.faceNormals);
    
    std::cout << "Detected " << creaseEdges.size() << " crease edges" << std::endl;
    
    if (creaseEdges.size() > 0)
    {
        std::cout << "PASS: Crease edges detected" << std::endl;
        
        // 打印前几条裂隙边的信息
        int count = 0;
        for (const auto& [key, info] : creaseEdges)
        {
            if (count++ >= 3) break;
            std::cout << "  Edge (" << key.v0 << ", " << key.v1 << "): "
                      << "max_gap = " << info.max_gap << std::endl;
        }
    }
    else
    {
        std::cout << "WARNING: No crease edges detected (may be expected for some models)" << std::endl;
    }
}

// ============================================================
// 测试 3: c_sharp 计算与缓存
// ============================================================
void testCSharpComputation(const std::string& nsmPath)
{
    std::cout << "\n=== Test 3: c_sharp Computation ===" << std::endl;
    
    // 加载 NSM 文件
    auto meshData = MeshBinaryLoader::loadFromNSM(nsmPath);
    
    if (meshData.vertices.empty())
    {
        std::cout << "FAIL: Could not load NSM file" << std::endl;
        return;
    }
    
    // 删除可能存在的缓存
    std::string engPath = NagataEnhanced::getEngFilepath(nsmPath);
    if (std::filesystem::exists(engPath))
    {
        std::filesystem::remove(engPath);
        std::cout << "Removed existing cache: " << engPath << std::endl;
    }
    
    // 计算并保存
    std::cout << "Computing c_sharp..." << std::endl;
    auto data = NagataEnhanced::computeOrLoadEnhancedData(
        meshData.vertices, meshData.faces, meshData.faceNormals, nsmPath, true);
    
    std::cout << "Computed " << data.size() << " c_sharp coefficients" << std::endl;
    
    // 验证缓存文件
    if (std::filesystem::exists(engPath))
    {
        std::cout << "PASS: Cache file created" << std::endl;
        
        // 测试从缓存加载
        NagataEnhanced::EnhancedNagataData cachedData;
        if (NagataEnhanced::loadEnhancedData(engPath, cachedData))
        {
            if (cachedData.size() == data.size())
            {
                std::cout << "PASS: Cache load successful, size matches" << std::endl;
            }
            else
            {
                std::cout << "FAIL: Cache size mismatch" << std::endl;
            }
        }
    }
    else
    {
        std::cout << "WARNING: Cache file not created (may be expected if no crease edges)" << std::endl;
    }
    
    // 打印一些 c_sharp 值
    int count = 0;
    for (const auto& [key, c] : data.c_sharps)
    {
        if (count++ >= 3) break;
        std::cout << "  Edge (" << key.v0 << ", " << key.v1 << "): "
                  << "c_sharp = (" << c.x << ", " << c.y << ", " << c.z << ")" << std::endl;
    }
}

// ============================================================
// 测试 4: 增强曲面求值
// ============================================================
void testEnhancedSurfaceEvaluation()
{
    std::cout << "\n=== Test 4: Enhanced Surface Evaluation ===" << std::endl;
    
    // 创建一个简单的三角形
    glm::vec3 v0(0.0f, 0.0f, 0.0f);
    glm::vec3 v1(1.0f, 0.0f, 0.0f);
    glm::vec3 v2(0.5f, 1.0f, 0.0f);
    
    // 不同的法向量 (模拟裂隙边情况)
    glm::vec3 n0 = glm::normalize(glm::vec3(0.0f, -0.3f, 1.0f));
    glm::vec3 n1 = glm::normalize(glm::vec3(0.0f, -0.3f, 1.0f));
    glm::vec3 n2 = glm::normalize(glm::vec3(0.0f, 0.5f, 1.0f));
    
    NagataPatch::NagataPatchData patch(v0, v1, v2, n0, n1, n2);
    
    // 模拟 c_sharp 值
    glm::vec3 c_sharp_1(0.1f, 0.0f, 0.05f);
    glm::vec3 c_sharp_2(0.0f, 0.1f, 0.03f);
    glm::vec3 c_sharp_3(0.05f, 0.05f, 0.02f);
    
    std::array<bool, 3> isCrease = {true, true, false};
    
    // 在边界处测试
    std::cout << "Testing at edge boundaries:" << std::endl;
    
    // 边1 (v=0)
    glm::vec3 p_edge1_orig = NagataPatch::evaluateSurface(patch, 0.5f, 0.0f);
    glm::vec3 p_edge1_enh = NagataEnhanced::evaluateSurfaceEnhanced(
        patch, 0.5f, 0.0f, c_sharp_1, c_sharp_2, c_sharp_3, isCrease, 0.1f);
    
    std::cout << "  Edge 1 (u=0.5, v=0):" << std::endl;
    std::cout << "    Original: (" << p_edge1_orig.x << ", " << p_edge1_orig.y << ", " << p_edge1_orig.z << ")" << std::endl;
    std::cout << "    Enhanced: (" << p_edge1_enh.x << ", " << p_edge1_enh.y << ", " << p_edge1_enh.z << ")" << std::endl;
    
    // 内部点 (远离边界)
    glm::vec3 p_interior_orig = NagataPatch::evaluateSurface(patch, 0.5f, 0.25f);
    glm::vec3 p_interior_enh = NagataEnhanced::evaluateSurfaceEnhanced(
        patch, 0.5f, 0.25f, c_sharp_1, c_sharp_2, c_sharp_3, isCrease, 0.1f);
    
    std::cout << "  Interior (u=0.5, v=0.25):" << std::endl;
    std::cout << "    Original: (" << p_interior_orig.x << ", " << p_interior_orig.y << ", " << p_interior_orig.z << ")" << std::endl;
    std::cout << "    Enhanced: (" << p_interior_enh.x << ", " << p_interior_enh.y << ", " << p_interior_enh.z << ")" << std::endl;
    
    // 验证顶点处仍然通过
    glm::vec3 p_v0 = NagataEnhanced::evaluateSurfaceEnhanced(
        patch, 0.0f, 0.0f, c_sharp_1, c_sharp_2, c_sharp_3, isCrease, 0.1f);
    glm::vec3 p_v1 = NagataEnhanced::evaluateSurfaceEnhanced(
        patch, 1.0f, 0.0f, c_sharp_1, c_sharp_2, c_sharp_3, isCrease, 0.1f);
    glm::vec3 p_v2 = NagataEnhanced::evaluateSurfaceEnhanced(
        patch, 1.0f, 1.0f, c_sharp_1, c_sharp_2, c_sharp_3, isCrease, 0.1f);
    
    float err0 = glm::length(p_v0 - v0);
    float err1 = glm::length(p_v1 - v1);
    float err2 = glm::length(p_v2 - v2);
    
    std::cout << "  Vertex interpolation errors: " << err0 << ", " << err1 << ", " << err2 << std::endl;
    
    if (err0 < 1e-5f && err1 < 1e-5f && err2 < 1e-5f)
    {
        std::cout << "PASS: Enhanced surface still passes through vertices" << std::endl;
    }
    else
    {
        std::cout << "FAIL: Enhanced surface does not pass through vertices" << std::endl;
    }
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "Nagata Enhanced Module Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 测试 1: ENG 文件读写
    testEngFileIO();
    
    // 测试 4: 增强曲面求值 (不需要外部文件)
    testEnhancedSurfaceEvaluation();
    
    // 测试 2 & 3 需要 NSM 文件
    std::string nsmPath = "output/cone_two_faces.nsm";
    if (argc > 1)
    {
        nsmPath = argv[1];
    }
    
    if (std::filesystem::exists(nsmPath))
    {
        testCreaseEdgeDetection(nsmPath);
        testCSharpComputation(nsmPath);
    }
    else
    {
        std::cout << "\n=== Skipping NSM-dependent tests ===" << std::endl;
        std::cout << "NSM file not found: " << nsmPath << std::endl;
        std::cout << "Usage: " << argv[0] << " <nsm_file_path>" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Completed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
