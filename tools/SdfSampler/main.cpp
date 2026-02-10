#include <sdflib/ExactOctreeSdf.h>
#include <sdflib/OctreeSdf.h>
#include <sdflib/SdfFunction.h>
#include <sdflib/utils/BoundingBox.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>

using namespace sdflib;
using namespace std::chrono;

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <input_sdf_file> <output_raw_file> [grid_resolution] [options]" << std::endl;
    std::cerr << "  input_sdf_file   : Path to the SDF binary file" << std::endl;
    std::cerr << "  output_raw_file  : Path to output raw volume data (can be loaded by Python)" << std::endl;
    std::cerr << "  grid_resolution  : Grid resolution for sampling (default: 128)" << std::endl;
    std::cerr << "\nOptions:\n";
    std::cerr << "  --inset <float>  : Shift sampling inside bbox by <float> cells (default: 0.0)\n";
    std::cerr << "                    Example: --inset 0.5 samples at cell centers (no endpoints)\n";
    std::cerr << "  --query <x> <y> <z> : Query a single world-space point and print distance\n";
    std::cerr << "  --debug          : With --query, print Octree leaf/coefficient diagnostics\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int gridRes = 128;
    int argi = 3;
    if (argc > 3 && std::string(argv[3]).rfind("--", 0) != 0) {
        gridRes = std::atoi(argv[3]);
        argi = 4;
    }
    float inset = 0.0f;
    bool queryMode = false;
    bool debugQuery = false;
    glm::vec3 queryPoint(0.0f);
    for (; argi < argc; argi++) {
        const std::string arg = argv[argi];
        if (arg == "--inset" && argi + 1 < argc) {
            inset = std::stof(argv[++argi]);
        } else if (arg == "--query" && argi + 3 < argc) {
            queryMode = true;
            queryPoint.x = std::stof(argv[++argi]);
            queryPoint.y = std::stof(argv[++argi]);
            queryPoint.z = std::stof(argv[++argi]);
        } else if (arg == "--debug") {
            debugQuery = true;
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "Loading SDF from: " << inputPath << std::endl;
    
    // Load SDF file
    std::unique_ptr<SdfFunction> sdf = SdfFunction::loadFromFile(inputPath);
    if (!sdf) {
        std::cerr << "Error: Failed to load SDF file" << std::endl;
        return 1;
    }

    // Get bounding box
    BoundingBox box = sdf->getBoundingBox();
    std::cout << "Bounding Box: Min(" << box.min.x << ", " << box.min.y << ", " << box.min.z << ")" << std::endl;
    std::cout << "              Max(" << box.max.x << ", " << box.max.y << ", " << box.max.z << ")" << std::endl;
    if (queryMode) {
        float dist = sdf->getDistance(queryPoint);
        std::cout << "Query point: (" << queryPoint.x << ", " << queryPoint.y << ", " << queryPoint.z << ")" << std::endl;
        std::cout << "Distance: " << dist << std::endl;
        if (debugQuery) {
            const OctreeSdf* oct = dynamic_cast<const OctreeSdf*>(sdf.get());
            if (oct) {
                auto info = oct->debugQuery(queryPoint);
                std::cout << "Debug outsideStartGrid: " << (info.outsideStartGrid ? 1 : 0) << std::endl;
                std::cout << "Debug startArrayPos: (" << info.startArrayPos.x << ", " << info.startArrayPos.y << ", " << info.startArrayPos.z << ")" << std::endl;
                std::cout << "Debug leafNodeIndex: " << info.leafNodeIndex << std::endl;
                std::cout << "Debug coeffIndex: " << info.coeffIndex << std::endl;
                std::cout << "Debug leafMin: (" << info.leafMin.x << ", " << info.leafMin.y << ", " << info.leafMin.z << ")" << std::endl;
                std::cout << "Debug leafSize: " << info.leafSize << std::endl;
                std::cout << "Debug leafFrac: (" << info.fracPart.x << ", " << info.fracPart.y << ", " << info.fracPart.z << ")" << std::endl;
                const glm::vec3 corners[8] =
                {
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(1.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f),
                    glm::vec3(1.0f, 1.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f),
                    glm::vec3(1.0f, 0.0f, 1.0f),
                    glm::vec3(0.0f, 1.0f, 1.0f),
                    glm::vec3(1.0f, 1.0f, 1.0f)
                };
                for (int i = 0; i < 8; i++) {
                    glm::vec3 wp = info.leafMin + info.leafSize * corners[i];
                    std::cout << "Debug corner" << i << " world: ("
                              << wp.x << ", " << wp.y << ", " << wp.z << ")"
                              << " val: " << info.cornerValues[i] << std::endl;
                }
            } else {
                std::cout << "Debug: loaded SDF is not an OctreeSdf" << std::endl;
            }
        }
        return 0;
    }

    std::cout << "Grid Resolution: " << gridRes << "x" << gridRes << "x" << gridRes << std::endl;
    std::cout << "Inset: " << inset << std::endl;

    // Sample SDF on uniform grid
    std::vector<float> gridData(gridRes * gridRes * gridRes);
    
    glm::vec3 boxSize = box.getSize();
    const float denom = static_cast<float>(gridRes - 1) + 2.0f * inset;
    float cellSizeX = boxSize.x / denom;
    float cellSizeY = boxSize.y / denom;
    float cellSizeZ = boxSize.z / denom;

    std::cout << "Sampling SDF..." << std::endl;
    auto startTime = high_resolution_clock::now();
    
    // Single-threaded sampling (SDF may not be thread-safe)
    int totalPoints = gridRes * gridRes * gridRes;
    int processedPoints = 0;
    int lastPercent = -1;
    
    for (int z = 0; z < gridRes; ++z) {
        for (int y = 0; y < gridRes; ++y) {
            for (int x = 0; x < gridRes; ++x) {
                glm::vec3 point(
                    box.min.x + (static_cast<float>(x) + inset) * cellSizeX,
                    box.min.y + (static_cast<float>(y) + inset) * cellSizeY,
                    box.min.z + (static_cast<float>(z) + inset) * cellSizeZ
                );
                float dist = sdf->getDistance(point);
                gridData[z * gridRes * gridRes + y * gridRes + x] = dist;
                processedPoints++;
            }
        }
        int percent = (z * 100) / gridRes;
        if (percent != lastPercent && percent % 10 == 0) {
            std::cout << "  Progress: " << percent << "%" << std::endl;
            lastPercent = percent;
        }
    }
    
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);
    std::cout << "Sampling completed in " << duration.count() / 1000.0 << " seconds" << std::endl;

    // Save to raw binary file with header
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open output file" << std::endl;
        return 1;
    }

    // Write header: gridRes (int), bbox min (3 floats), bbox max (3 floats)
    outFile.write(reinterpret_cast<const char*>(&gridRes), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&box.min.x), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&box.min.y), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&box.min.z), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&box.max.x), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&box.max.y), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&box.max.z), sizeof(float));
    
    // Write grid data
    outFile.write(reinterpret_cast<const char*>(gridData.data()), gridData.size() * sizeof(float));
    outFile.close();

    std::cout << "Output saved to: " << outputPath << std::endl;
    std::cout << "Total samples: " << gridData.size() << std::endl;

    return 0;
}
