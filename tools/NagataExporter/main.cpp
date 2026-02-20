#include <sdflib/utils/MeshBinaryLoader.h>
#include <sdflib/utils/NagataEnhanced.h>
#include <sdflib/utils/NagataPatch.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <input.nsm> <output_dir> <subdivision_level> [tolerance]\n";
}

static bool normalizeSafe(glm::vec3& v)
{
    float len = glm::length(v);
    if (len < 1e-12f)
    {
        return false;
    }
    v /= len;
    return true;
}

static int resolutionForLevel(int level)
{
    if (level < 0)
    {
        level = 0;
    }
    return (1 << level) + 1;
}

static void samplePatch(
    const sdflib::NagataPatch::NagataPatchData& patch,
    const sdflib::NagataPatch::PatchEnhancementData& enhance,
    int resolution,
    std::vector<glm::vec3>& vertices,
    std::vector<std::array<uint32_t, 3>>& faces)
{
    std::vector<int> indexMap(resolution * resolution, -1);

    for (int i = 0; i < resolution; ++i)
    {
        float u = (resolution == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(resolution - 1);
        for (int j = 0; j < resolution; ++j)
        {
            float v = (resolution == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(resolution - 1);
            if (v > u + 1e-8f)
            {
                continue;
            }
            glm::vec3 p = sdflib::NagataPatch::evaluateSurface(patch, enhance, u, v);
            int idx = static_cast<int>(vertices.size());
            vertices.push_back(p);
            indexMap[i * resolution + j] = idx;
        }
    }

    for (int i = 0; i < resolution - 1; ++i)
    {
        for (int j = 0; j < resolution - 1; ++j)
        {
            int i00 = indexMap[i * resolution + j];
            int i10 = indexMap[(i + 1) * resolution + j];
            int i11 = indexMap[(i + 1) * resolution + (j + 1)];
            int i01 = indexMap[i * resolution + (j + 1)];

            if (i00 >= 0 && i10 >= 0 && i11 >= 0)
            {
                faces.push_back({static_cast<uint32_t>(i00), static_cast<uint32_t>(i10), static_cast<uint32_t>(i11)});
            }
            if (i00 >= 0 && i11 >= 0 && i01 >= 0)
            {
                faces.push_back({static_cast<uint32_t>(i00), static_cast<uint32_t>(i11), static_cast<uint32_t>(i01)});
            }
        }
    }
}

static std::vector<glm::vec3> computeVertexNormals(
    const std::vector<glm::vec3>& vertices,
    const std::vector<std::array<uint32_t, 3>>& faces)
{
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0f));
    for (const auto& f : faces)
    {
        const glm::vec3& v0 = vertices[f[0]];
        const glm::vec3& v1 = vertices[f[1]];
        const glm::vec3& v2 = vertices[f[2]];
        glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
        if (!normalizeSafe(n))
        {
            continue;
        }
        normals[f[0]] += n;
        normals[f[1]] += n;
        normals[f[2]] += n;
    }
    for (auto& n : normals)
    {
        if (!normalizeSafe(n))
        {
            n = glm::vec3(0.0f, 0.0f, 1.0f);
        }
    }
    return normals;
}

static bool writeObj(
    const fs::path& outputPath,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec3>& normals,
    const std::vector<std::array<uint32_t, 3>>& faces)
{
    std::ofstream out(outputPath);
    if (!out.is_open())
    {
        return false;
    }

    out << std::setprecision(16);

    for (const auto& v : vertices)
    {
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (const auto& n : normals)
    {
        out << "vn " << n.x << " " << n.y << " " << n.z << "\n";
    }
    for (const auto& f : faces)
    {
        uint32_t a = f[0] + 1;
        uint32_t b = f[1] + 1;
        uint32_t c = f[2] + 1;
        out << "f " << a << "//" << a << " " << b << "//" << b << " " << c << "//" << c << "\n";
    }
    return true;
}

int exportEnhancedNagataSubdivision(
    const std::string& inputPath,
    const std::string& outputPath,
    int subdivisionLevel,
    double tolerance)
{
    auto meshData = sdflib::MeshBinaryLoader::loadFromNSM(inputPath);
    if (meshData.vertices.empty())
    {
        SPDLOG_ERROR("Failed to load NSM data");
        return 1;
    }

    std::vector<sdflib::NagataPatch::NagataPatchData> patches = sdflib::MeshBinaryLoader::createNagataPatchData(meshData);
    auto creaseEdges = sdflib::NagataEnhanced::detectCreaseEdges(
        meshData.vertices, meshData.faces, meshData.faceNormals);
    sdflib::NagataEnhanced::EnhancedNagataData enhancedData =
        sdflib::NagataEnhanced::computeCSharpForEdges(
            creaseEdges, meshData.vertices, meshData.faces, meshData.faceNormals,
            static_cast<float>(tolerance));

    std::vector<sdflib::NagataPatch::PatchEnhancementData> enhanced(patches.size());
    float kFactor = static_cast<float>(tolerance);

    for (size_t i = 0; i < meshData.faces.size() && i < enhanced.size(); ++i)
    {
        const auto& face = meshData.faces[i];
        auto& pe = enhanced[i];

        auto setEdge = [&](int edgeIdx, uint32_t a, uint32_t b)
        {
            const sdflib::NagataEnhanced::EdgeKey key(a, b);
            if (!enhancedData.hasEdge(key))
            {
                return;
            }
            pe.edges[edgeIdx].enabled = true;
            pe.edges[edgeIdx].c_sharp = enhancedData.getCSharp(key);
            pe.edges[edgeIdx].k_factor = kFactor;
        };

        setEdge(0, face[0], face[1]);
        setEdge(1, face[1], face[2]);
        setEdge(2, face[0], face[2]);
    }

    int resolution = resolutionForLevel(subdivisionLevel);

    std::vector<glm::vec3> vertices;
    std::vector<std::array<uint32_t, 3>> faces;

    vertices.reserve(patches.size() * resolution * resolution / 2);

    for (size_t i = 0; i < patches.size(); ++i)
    {
        samplePatch(patches[i], enhanced[i], resolution, vertices, faces);
    }

    std::vector<glm::vec3> normals = computeVertexNormals(vertices, faces);

    fs::path outputDir = fs::path(outputPath);
    fs::path outputFile;
    if (fs::is_directory(outputDir))
    {
        std::string stem = fs::path(inputPath).stem().string();
        outputFile = outputDir / (stem + "_enhanced_L" + std::to_string(subdivisionLevel) + ".obj");
    }
    else
    {
        fs::path parent = outputDir.parent_path();
        if (parent.empty())
        {
            parent = fs::current_path();
        }
        std::string stem = fs::path(inputPath).stem().string();
        outputFile = parent / (stem + "_enhanced_L" + std::to_string(subdivisionLevel) + ".obj");
    }

    if (!writeObj(outputFile, vertices, normals, faces))
    {
        SPDLOG_ERROR("Failed to write OBJ: {}", outputFile.string());
        return 1;
    }

    SPDLOG_INFO("Exported OBJ: {}", outputFile.string());
    SPDLOG_INFO("Vertices: {}, Faces: {}", vertices.size(), faces.size());
    return 0;
}

int main(int argc, char** argv)
{
    auto logger = spdlog::stdout_color_mt("NagataExporter");
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    if (argc < 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int level = std::stoi(argv[3]);
    double tolerance = 0.1;
    if (argc >= 5)
    {
        tolerance = std::stod(argv[4]);
    }

    return exportEnhancedNagataSubdivision(inputPath, outputPath, level, tolerance);
}
