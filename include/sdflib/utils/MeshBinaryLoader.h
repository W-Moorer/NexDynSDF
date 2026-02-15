#ifndef MESH_BINARY_LOADER_H
#define MESH_BINARY_LOADER_H

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <cstdint>
#include "NagataPatch.h"
#include "Mesh.h"

namespace sdflib
{
namespace MeshBinaryLoader
{
    /**
     * @brief Mesh data structure
     * 
     * Stores mesh data loaded from binary files
     */
    struct MeshData
    {
        std::vector<glm::vec3> vertices;                    // Vertex coordinates
        std::vector<std::array<uint32_t, 3>> faces;        // Face indices
        std::vector<std::array<glm::vec3, 3>> faceNormals; // Normals for each vertex of each face
        
        /**
         * @brief Get number of vertices
         */
        size_t getNumVertices() const { return vertices.size(); }
        
        /**
         * @brief Get number of faces
         */
        size_t getNumFaces() const { return faces.size(); }
        
        /**
         * @brief Get vertex coordinates of a face
         * 
         * @param faceIndex Face index
         * @return std::array<glm::vec3, 3> Coordinates of the three vertices
         */
        std::array<glm::vec3, 3> getFaceVertices(size_t faceIndex) const
        {
            std::array<glm::vec3, 3> result;
            for (int i = 0; i < 3; ++i)
            {
                result[i] = vertices[faces[faceIndex][i]];
            }
            return result;
        }
        
        /**
         * @brief Get vertex normals of a face
         * 
         * @param faceIndex Face index
         * @return std::array<glm::vec3, 3> Three vertex normals in the face
         */
        std::array<glm::vec3, 3> getFaceVertexNormals(size_t faceIndex) const
        {
            return faceNormals[faceIndex];
        }
        
        /**
         * @brief Compute bounding box
         * 
         * @param minBound Output: min bound
         * @param maxBound Output: max bound
         */
        void computeBounds(glm::vec3& minBound, glm::vec3& maxBound) const
        {
            if (vertices.empty())
            {
                minBound = glm::vec3(0.0f);
                maxBound = glm::vec3(0.0f);
                return;
            }
            
            minBound = vertices[0];
            maxBound = vertices[0];
            
            for (const auto& v : vertices)
            {
                minBound = glm::min(minBound, v);
                maxBound = glm::max(maxBound, v);
            }
        }
    };

    struct MeshDataDouble
    {
        std::vector<glm::dvec3> vertices;
        std::vector<std::array<uint32_t, 3>> faces;
        std::vector<std::array<glm::dvec3, 3>> faceNormals;
        
        size_t getNumVertices() const { return vertices.size(); }
        size_t getNumFaces() const { return faces.size(); }
        
        std::array<glm::dvec3, 3> getFaceVertices(size_t faceIndex) const
        {
            std::array<glm::dvec3, 3> result;
            for (int i = 0; i < 3; ++i)
            {
                result[i] = vertices[faces[faceIndex][i]];
            }
            return result;
        }
        
        std::array<glm::dvec3, 3> getFaceVertexNormals(size_t faceIndex) const
        {
            return faceNormals[faceIndex];
        }
        
        void computeBounds(glm::dvec3& minBound, glm::dvec3& maxBound) const
        {
            if (vertices.empty())
            {
                minBound = glm::dvec3(0.0);
                maxBound = glm::dvec3(0.0);
                return;
            }
            
            minBound = vertices[0];
            maxBound = vertices[0];
            
            for (const auto& v : vertices)
            {
                minBound = glm::min(minBound, v);
                maxBound = glm::max(maxBound, v);
            }
        }
    };
    
    /**
     * @brief Load mesh data from binary file
     * 
     * File format:
     *     - 4 bytes: numVertices (uint32)
     *     - 4 bytes: numFaces (uint32)
     *     - N*12 bytes: vertices (float32 * 3 * N)
     *     - M*12 bytes: faces (uint32 * 3 * M)
     *     - M*36 bytes: face normals (float32 * 3 * 3 * M)
     * 
     * @param filepath binary file path
     * @return MeshData loaded mesh data
     */
    inline MeshData loadFromBinary(const std::string& filepath)
    {
        MeshData meshData;
        
        // Open file
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "MeshBinaryLoader: Cannot open file " << filepath << std::endl;
            return meshData;
        }
        
        // Read header
        uint32_t numVertices = 0;
        uint32_t numFaces = 0;
        
        file.read(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&numFaces), sizeof(uint32_t));
        
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: Failed to read header " << filepath << std::endl;
            return meshData;
        }
        
        // Allocate memory
        meshData.vertices.resize(numVertices);
        meshData.faces.resize(numFaces);
        meshData.faceNormals.resize(numFaces);
        
        // Read vertices
        for (uint32_t i = 0; i < numVertices; ++i)
        {
            float x, y, z;
            file.read(reinterpret_cast<char*>(&x), sizeof(float));
            file.read(reinterpret_cast<char*>(&y), sizeof(float));
            file.read(reinterpret_cast<char*>(&z), sizeof(float));
            meshData.vertices[i] = glm::vec3(x, y, z);
        }
        
        // Read faces
        for (uint32_t i = 0; i < numFaces; ++i)
        {
            uint32_t i0, i1, i2;
            file.read(reinterpret_cast<char*>(&i0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i1), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i2), sizeof(uint32_t));
            meshData.faces[i] = {i0, i1, i2};
        }
        
        // Read face normals
        for (uint32_t i = 0; i < numFaces; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                float nx, ny, nz;
                file.read(reinterpret_cast<char*>(&nx), sizeof(float));
                file.read(reinterpret_cast<char*>(&ny), sizeof(float));
                file.read(reinterpret_cast<char*>(&nz), sizeof(float));
                meshData.faceNormals[i][j] = glm::vec3(nx, ny, nz);
            }
        }
        
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: Failed to read data " << filepath << std::endl;
            meshData.vertices.clear();
            meshData.faces.clear();
            meshData.faceNormals.clear();
            return meshData;
        }
        
        file.close();
        
        std::cout << "MeshBinaryLoader: Successfully loaded " << filepath << std::endl;
        std::cout << "  Vertices: " << numVertices << std::endl;
        std::cout << "  Faces: " << numFaces << std::endl;
        
        return meshData;
    }
    
    /**
     * @brief Create NagataPatchData array from mesh data
     * 
     * Uses loaded mesh data to create Nagata patches for interpolation
     * 
     * @param meshData mesh data
     * @return std::vector<NagataPatch::NagataPatchData> Nagata patches
     */
    inline std::vector<NagataPatch::NagataPatchData> createNagataPatchData(
        const MeshData& meshData)
    {
        std::vector<NagataPatch::NagataPatchData> nagataPatches;
        nagataPatches.reserve(meshData.getNumFaces());
        
        for (size_t i = 0; i < meshData.getNumFaces(); ++i)
        {
            // Get face vertex coordinates
            auto faceVertices = meshData.getFaceVertices(i);
            
            // Get face vertex normals
            auto faceNormals = meshData.getFaceVertexNormals(i);
            
            // Create Nagata patch data
            nagataPatches.emplace_back(
                faceVertices[0], faceVertices[1], faceVertices[2],
                faceNormals[0], faceNormals[1], faceNormals[2]
            );
        }
        
        return nagataPatches;
    }

    /**
     * @brief Create NagataPatchData array from Mesh object
     * @param mesh Mesh object
     * @param outPatches Output Nagata patches
     */
    inline void createNagataPatchData(
        const Mesh& mesh,
        std::vector<NagataPatch::NagataPatchData>& outPatches)
    {
        const auto& vertices = mesh.getVertices();
        const auto& indices = mesh.getIndices();
        const auto& normals = mesh.getNormals();
        
        size_t numTriangles = indices.size() / 3;
        outPatches.clear();
        outPatches.reserve(numTriangles);
        
        bool hasNormals = !normals.empty();
        if (!hasNormals)
        {
            std::cerr << "MeshBinaryLoader: Mesh has no normals! Nagata patches will have zero normals." << std::endl;
        }

        for (size_t i = 0; i < numTriangles; ++i)
        {
            uint32_t idx0 = indices[i * 3];
            uint32_t idx1 = indices[i * 3 + 1];
            uint32_t idx2 = indices[i * 3 + 2];
            
            glm::vec3 v0 = vertices[idx0];
            glm::vec3 v1 = vertices[idx1];
            glm::vec3 v2 = vertices[idx2];
            
            glm::vec3 n0 = hasNormals ? normals[idx0] : glm::vec3(0.0f);
            glm::vec3 n1 = hasNormals ? normals[idx1] : glm::vec3(0.0f);
            glm::vec3 n2 = hasNormals ? normals[idx2] : glm::vec3(0.0f);
            
            // If explicit face normals are not available (Mesh only has vertex normals),
            // this uses smoothed vertex normals.
            outPatches.emplace_back(v0, v1, v2, n0, n1, n2);
        }
    }
    
    /**
     * @brief Load mesh data from NSM file
     * 
     * NSM file format:
     *     - 64 bytes header:
     *         - 4 bytes: Magic "NSM\0"
     *         - 4 bytes: Version (uint32)
     *         - 4 bytes: NumVertices (uint32)
     *         - 4 bytes: NumTriangles (uint32)
     *         - 48 bytes: Reserved
     *     - N*24 bytes: vertices (double * 3 * N)
     *     - M*12 bytes: triangles (uint32 * 3 * M)
     *     - M*4 bytes: face IDs (uint32 * M)
     *     - M*72 bytes: vertex normals (double * 3 * 3 * M)
     */
    inline MeshData loadFromNSM(const std::string& filepath)
    {
        MeshData meshData;
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "MeshBinaryLoader: Cannot open NSM file " << filepath << std::endl;
            return meshData;
        }
        
        // Read header (64 bytes)
        char header[64];
        file.read(header, 64);
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: NSM header incomplete" << std::endl;
            return meshData;
        }
        
        // Validate magic
        if (header[0] != 'N' || header[1] != 'S' || header[2] != 'M' || header[3] != '\0')
        {
            std::cerr << "MeshBinaryLoader: Invalid NSM magic" << std::endl;
            return meshData;
        }
        
        // Parse header
        uint32_t version = *reinterpret_cast<uint32_t*>(header + 4);
        uint32_t numVertices = *reinterpret_cast<uint32_t*>(header + 8);
        uint32_t numTriangles = *reinterpret_cast<uint32_t*>(header + 12);
        
        if (version != 1)
        {
            std::cerr << "MeshBinaryLoader: Unsupported NSM version " << version << std::endl;
            return meshData;
        }
        
        // Allocate memory
        meshData.vertices.resize(numVertices);
        meshData.faces.resize(numTriangles);
        meshData.faceNormals.resize(numTriangles);
        
        // Read vertices (double)
        for (uint32_t i = 0; i < numVertices; ++i)
        {
            double x, y, z;
            file.read(reinterpret_cast<char*>(&x), sizeof(double));
            file.read(reinterpret_cast<char*>(&y), sizeof(double));
            file.read(reinterpret_cast<char*>(&z), sizeof(double));
            meshData.vertices[i] = glm::vec3(static_cast<float>(x), 
                                              static_cast<float>(y), 
                                              static_cast<float>(z));
        }
        
        // Read faces
        for (uint32_t i = 0; i < numTriangles; ++i)
        {
            uint32_t i0, i1, i2;
            file.read(reinterpret_cast<char*>(&i0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i1), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i2), sizeof(uint32_t));
            meshData.faces[i] = {i0, i1, i2};
        }
        
        // Skip face IDs
        file.seekg(numTriangles * sizeof(uint32_t), std::ios::cur);
        
        // Read vertex normals (double)
        for (uint32_t i = 0; i < numTriangles; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                double nx, ny, nz;
                file.read(reinterpret_cast<char*>(&nx), sizeof(double));
                file.read(reinterpret_cast<char*>(&ny), sizeof(double));
                file.read(reinterpret_cast<char*>(&nz), sizeof(double));
                meshData.faceNormals[i][j] = glm::vec3(static_cast<float>(nx),
                                                        static_cast<float>(ny),
                                                        static_cast<float>(nz));
            }
        }
        
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: NSM data read failed" << std::endl;
            meshData.vertices.clear();
            meshData.faces.clear();
            meshData.faceNormals.clear();
            return meshData;
        }
        
        std::cout << "MeshBinaryLoader: Successfully loaded NSM file " << filepath << std::endl;
        std::cout << "  Vertices: " << numVertices << std::endl;
        std::cout << "  Triangles: " << numTriangles << std::endl;
        
        return meshData;
    }

    inline MeshDataDouble loadFromNSMDouble(const std::string& filepath)
    {
        MeshDataDouble meshData;
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "MeshBinaryLoader: Cannot open NSM file " << filepath << std::endl;
            return meshData;
        }
        
        char header[64];
        file.read(header, 64);
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: NSM header incomplete" << std::endl;
            return meshData;
        }
        
        if (header[0] != 'N' || header[1] != 'S' || header[2] != 'M' || header[3] != '\0')
        {
            std::cerr << "MeshBinaryLoader: Invalid NSM magic" << std::endl;
            return meshData;
        }
        
        uint32_t version = *reinterpret_cast<uint32_t*>(header + 4);
        uint32_t numVertices = *reinterpret_cast<uint32_t*>(header + 8);
        uint32_t numTriangles = *reinterpret_cast<uint32_t*>(header + 12);
        
        if (version != 1)
        {
            std::cerr << "MeshBinaryLoader: Unsupported NSM version " << version << std::endl;
            return meshData;
        }
        
        meshData.vertices.resize(numVertices);
        meshData.faces.resize(numTriangles);
        meshData.faceNormals.resize(numTriangles);
        
        for (uint32_t i = 0; i < numVertices; ++i)
        {
            double x, y, z;
            file.read(reinterpret_cast<char*>(&x), sizeof(double));
            file.read(reinterpret_cast<char*>(&y), sizeof(double));
            file.read(reinterpret_cast<char*>(&z), sizeof(double));
            meshData.vertices[i] = glm::dvec3(x, y, z);
        }
        
        for (uint32_t i = 0; i < numTriangles; ++i)
        {
            uint32_t i0, i1, i2;
            file.read(reinterpret_cast<char*>(&i0), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i1), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&i2), sizeof(uint32_t));
            meshData.faces[i] = {i0, i1, i2};
        }
        
        file.seekg(numTriangles * sizeof(uint32_t), std::ios::cur);
        
        for (uint32_t i = 0; i < numTriangles; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                double nx, ny, nz;
                file.read(reinterpret_cast<char*>(&nx), sizeof(double));
                file.read(reinterpret_cast<char*>(&ny), sizeof(double));
                file.read(reinterpret_cast<char*>(&nz), sizeof(double));
                meshData.faceNormals[i][j] = glm::dvec3(nx, ny, nz);
            }
        }
        
        if (!file)
        {
            std::cerr << "MeshBinaryLoader: NSM data read failed" << std::endl;
            meshData.vertices.clear();
            meshData.faces.clear();
            meshData.faceNormals.clear();
            return meshData;
        }
        
        std::cout << "MeshBinaryLoader: Successfully loaded NSM file " << filepath << std::endl;
        std::cout << "  Vertices: " << numVertices << std::endl;
        std::cout << "  Triangles: " << numTriangles << std::endl;
        
        return meshData;
    }
}
}

#endif // MESH_BINARY_LOADER_H
