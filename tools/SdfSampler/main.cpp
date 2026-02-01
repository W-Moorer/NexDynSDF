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
    std::cerr << "Usage: " << programName << " <input_sdf_file> <output_raw_file> [grid_resolution]" << std::endl;
    std::cerr << "  input_sdf_file   : Path to the SDF binary file" << std::endl;
    std::cerr << "  output_raw_file  : Path to output raw volume data (can be loaded by Python)" << std::endl;
    std::cerr << "  grid_resolution  : Grid resolution for sampling (default: 128)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int gridRes = (argc > 3) ? std::atoi(argv[3]) : 128;

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
    std::cout << "Grid Resolution: " << gridRes << "x" << gridRes << "x" << gridRes << std::endl;

    // Sample SDF on uniform grid
    std::vector<float> gridData(gridRes * gridRes * gridRes);
    
    glm::vec3 boxSize = box.getSize();
    float cellSizeX = boxSize.x / (gridRes - 1);
    float cellSizeY = boxSize.y / (gridRes - 1);
    float cellSizeZ = boxSize.z / (gridRes - 1);

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
                    box.min.x + x * cellSizeX,
                    box.min.y + y * cellSizeY,
                    box.min.z + z * cellSizeZ
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
