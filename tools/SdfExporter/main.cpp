/**
 * @file main.cpp
 * @brief SDF Exporter tool - converts mesh files to SDF
 */

#include <sdflib/SdfFunction.h>
#include <sdflib/OctreeSdf.h>
#include <sdflib/ExactOctreeSdf.h>
#include <sdflib/utils/Mesh.h>
#include <sdflib/utils/Timer.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <input.vtp> <output.bin> [options]\n"
              << "\nOptions:\n"
              << "  --depth <n>              Octree depth (default: 8)\n"
              << "  --start_depth <n>        Start depth (default: 1)\n"
              << "  --algorithm <type>       Algorithm: continuity, no_continuity, uniform (default: continuity)\n"
              << "  --sdf_format <format>    SDF format: octree, exact_octree (default: octree)\n"
              << "  --termination <threshold> Termination threshold (default: 1e-3)\n"
              << "  --num_threads <n>        Number of threads (default: 1)\n"
              << "  --help                   Show this help message\n";
}

struct Options
{
    std::string inputFile;
    std::string outputFile;
    uint32_t depth = 8;
    uint32_t startDepth = 1;
    sdflib::OctreeSdf::InitAlgorithm algorithm = sdflib::OctreeSdf::InitAlgorithm::CONTINUITY;
    std::string sdfFormat = "octree";
    float terminationThreshold = 1e-3f;
    uint32_t numThreads = 1;
};

bool parseArgs(int argc, char** argv, Options& opts)
{
    if (argc < 3)
    {
        return false;
    }

    opts.inputFile = argv[1];
    opts.outputFile = argv[2];

    for (int i = 3; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            return false;
        }
        else if (arg == "--depth" && i + 1 < argc)
        {
            opts.depth = std::stoi(argv[++i]);
        }
        else if (arg == "--start_depth" && i + 1 < argc)
        {
            opts.startDepth = std::stoi(argv[++i]);
        }
        else if (arg == "--algorithm" && i + 1 < argc)
        {
            std::string algo = argv[++i];
            if (algo == "continuity")
                opts.algorithm = sdflib::OctreeSdf::InitAlgorithm::CONTINUITY;
            else if (algo == "no_continuity")
                opts.algorithm = sdflib::OctreeSdf::InitAlgorithm::NO_CONTINUITY;
            else if (algo == "uniform")
                opts.algorithm = sdflib::OctreeSdf::InitAlgorithm::UNIFORM;
            else
            {
                SPDLOG_ERROR("Unknown algorithm: {}", algo);
                return false;
            }
        }
        else if (arg == "--sdf_format" && i + 1 < argc)
        {
            opts.sdfFormat = argv[++i];
        }
        else if (arg == "--termination" && i + 1 < argc)
        {
            opts.terminationThreshold = std::stof(argv[++i]);
        }
        else if (arg == "--num_threads" && i + 1 < argc)
        {
            opts.numThreads = std::stoi(argv[++i]);
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    // Setup logging
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    Options opts;
    if (!parseArgs(argc, argv, opts))
    {
        printUsage(argv[0]);
        return 1;
    }

    // Check input file exists
    if (!fs::exists(opts.inputFile))
    {
        SPDLOG_ERROR("Input file does not exist: {}", opts.inputFile);
        return 1;
    }

    // Create output directory if needed
    fs::path outputPath(opts.outputFile);
    if (!outputPath.parent_path().empty() && !fs::exists(outputPath.parent_path()))
    {
        fs::create_directories(outputPath.parent_path());
    }

    SPDLOG_INFO("============================================");
    SPDLOG_INFO("SDF Exporter");
    SPDLOG_INFO("============================================");
    SPDLOG_INFO("Input file: {}", opts.inputFile);
    SPDLOG_INFO("Output file: {}", opts.outputFile);
    SPDLOG_INFO("Depth: {}", opts.depth);
    SPDLOG_INFO("Start depth: {}", opts.startDepth);
    SPDLOG_INFO("SDF format: {}", opts.sdfFormat);
    SPDLOG_INFO("Termination threshold: {}", opts.terminationThreshold);
    SPDLOG_INFO("Threads: {}", opts.numThreads);

    // Load mesh
    sdflib::Timer timer;
    SPDLOG_INFO("Loading mesh...");

    sdflib::Mesh mesh(opts.inputFile);
    if (mesh.getVertices().empty())
    {
        SPDLOG_ERROR("Failed to load mesh");
        return 1;
    }

    SPDLOG_INFO("Mesh loaded in {} ms", timer.getElapsedMilliseconds());

    // Get bounding box with margin
    sdflib::BoundingBox box = mesh.getBoundingBox();
    box.addMargin(0.1f * box.getSize().x); // Add 10% margin

    // Build SDF
    timer.start();
    std::unique_ptr<sdflib::SdfFunction> sdf;

    if (opts.sdfFormat == "octree")
    {
        SPDLOG_INFO("Building Octree SDF...");
        sdf = std::make_unique<sdflib::OctreeSdf>(
            mesh, box, opts.depth, opts.startDepth,
            opts.terminationThreshold, opts.algorithm, opts.numThreads);
    }
    else if (opts.sdfFormat == "exact_octree")
    {
        SPDLOG_INFO("Building Exact Octree SDF...");
        sdf = std::make_unique<sdflib::ExactOctreeSdf>(
            mesh, box, opts.depth, opts.startDepth, 32, opts.numThreads);
    }
    else
    {
        SPDLOG_ERROR("Unknown SDF format: {}", opts.sdfFormat);
        return 1;
    }

    SPDLOG_INFO("SDF built in {} ms", timer.getElapsedMilliseconds());

    // Save SDF
    timer.start();
    SPDLOG_INFO("Saving SDF to file...");

    if (!sdf->saveToFile(opts.outputFile))
    {
        SPDLOG_ERROR("Failed to save SDF");
        return 1;
    }

    // Get file size
    auto fileSize = fs::file_size(opts.outputFile);
    SPDLOG_INFO("SDF saved in {} ms", timer.getElapsedMilliseconds());
    SPDLOG_INFO("File size: {} bytes ({:.2f} MB)", fileSize, fileSize / (1024.0 * 1024.0));

    // Test query
    glm::vec3 testPoint = box.getCenter();
    float dist = sdf->getDistance(testPoint);
    SPDLOG_INFO("Test query at center: distance = {}", dist);

    SPDLOG_INFO("============================================");
    SPDLOG_INFO("Export complete!");
    SPDLOG_INFO("============================================");

    return 0;
}
