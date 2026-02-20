#include <sdflib/utils/NagataEnhanced.h>
#include <sdflib/utils/MeshBinaryLoader.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <input.nsm> <output.eng>\n"
              << "  <input.nsm>   NSM mesh file\n"
              << "  <output.eng>  Output ENG file\n";
}

int main(int argc, char** argv)
{
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    if (argc < 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    if (!fs::exists(inputFile))
    {
        SPDLOG_ERROR("Input file does not exist: {}", inputFile);
        return 1;
    }

    std::string ext = fs::path(inputFile).extension().string();
    if (ext != ".nsm")
    {
        SPDLOG_ERROR("Only .nsm input is supported");
        return 1;
    }

    fs::path outputPath(outputFile);
    if (!outputPath.parent_path().empty() && !fs::exists(outputPath.parent_path()))
    {
        fs::create_directories(outputPath.parent_path());
    }

    SPDLOG_INFO("============================================");
    SPDLOG_INFO("ENG Exporter");
    SPDLOG_INFO("============================================");
    SPDLOG_INFO("Input file: {}", inputFile);
    SPDLOG_INFO("Output file: {}", outputFile);

    SPDLOG_INFO("Loading NSM file...");
    auto meshData = sdflib::MeshBinaryLoader::loadFromNSM(inputFile);
    if (meshData.vertices.empty())
    {
        SPDLOG_ERROR("Failed to load NSM data");
        return 1;
    }

    SPDLOG_INFO("Computing enhanced Nagata data...");
    auto enhancedData = sdflib::NagataEnhanced::computeOrLoadEnhancedData(
        meshData.vertices, meshData.faces, meshData.faceNormals, inputFile, false);

    SPDLOG_INFO("Saving ENG to file...");
    if (!sdflib::NagataEnhanced::saveEnhancedData(outputFile, enhancedData))
    {
        SPDLOG_ERROR("Failed to save ENG");
        return 1;
    }

    auto fileSize = fs::file_size(outputFile);
    SPDLOG_INFO("ENG saved. File size: {} bytes ({:.2f} MB)", fileSize, fileSize / (1024.0 * 1024.0));
    SPDLOG_INFO("============================================");
    SPDLOG_INFO("Export complete!");
    SPDLOG_INFO("============================================");

    return 0;
}
