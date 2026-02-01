/**
 * @file Mesh.h
 * @brief Triangle mesh data structure with VTP file support
 */

#pragma once

#include "BoundingBox.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>

namespace sdflib
{

class Mesh
{
public:
    Mesh() = default;
    
    // Load from VTP file (VTK XML format)
    explicit Mesh(const std::string& filePath);
    
    // Create from raw data
    Mesh(glm::vec3* vertices, uint32_t numVertices,
         uint32_t* indices, uint32_t numIndices);

    // Getters
    const std::vector<glm::vec3>& getVertices() const { return mVertices; }
    const std::vector<uint32_t>& getIndices() const { return mIndices; }
    const std::vector<glm::vec3>& getNormals() const { return mNormals; }
    const BoundingBox& getBoundingBox() const { return mBBox; }
    
    std::vector<glm::vec3>& getVertices() { return mVertices; }
    std::vector<uint32_t>& getIndices() { return mIndices; }

    // Transform
    void applyTransform(glm::mat4 trans);
    
    // Compute normals from geometry
    void computeNormals();
    
    // Compute bounding box
    void computeBoundingBox();

private:
    std::vector<glm::vec3> mVertices;
    std::vector<uint32_t> mIndices;
    std::vector<glm::vec3> mNormals;
    BoundingBox mBBox;

    // Parse VTP XML format
    bool parseVTP(const std::string& filePath);
    std::vector<float> parseDataArray(const std::string& content, const std::string& arrayName);
    std::string extractXMLTag(const std::string& content, const std::string& tagName);
    
    // Parse OBJ format
    bool parseOBJ(const std::string& filePath);
};

}
