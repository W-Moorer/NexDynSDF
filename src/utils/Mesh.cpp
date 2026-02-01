/**
 * @file Mesh.cpp
 * @brief Triangle mesh implementation with VTP file support
 */

#include "sdflib/utils/Mesh.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <regex>

namespace sdflib
{

Mesh::Mesh(const std::string& filePath)
{
    // Determine file type by extension
    std::string ext = filePath.substr(filePath.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    bool success = false;
    if (ext == "vtp")
    {
        success = parseVTP(filePath);
    }
    else if (ext == "obj")
    {
        success = parseOBJ(filePath);
    }
    else
    {
        SPDLOG_ERROR("Unsupported file format: {}", ext);
        return;
    }
    
    if (!success)
    {
        SPDLOG_ERROR("Failed to load mesh from: {}", filePath);
    }
}

Mesh::Mesh(glm::vec3* vertices, uint32_t numVertices,
           uint32_t* indices, uint32_t numIndices)
{
    mVertices.resize(numVertices);
    std::memcpy(mVertices.data(), vertices, sizeof(glm::vec3) * numVertices);

    mIndices.resize(numIndices);
    std::memcpy(mIndices.data(), indices, sizeof(uint32_t) * numIndices);
    
    computeBoundingBox();
}

bool Mesh::parseVTP(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        SPDLOG_ERROR("Cannot open file: {}", filePath);
        return false;
    }

    // Read entire file content
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    // Check if it's a VTP file
    if (content.find("<VTKFile") == std::string::npos)
    {
        SPDLOG_ERROR("File is not a valid VTP file: {}", filePath);
        return false;
    }

    // Parse points (vertices)
    std::vector<float> points = parseDataArray(content, "Points");
    if (points.empty())
    {
        SPDLOG_ERROR("No points found in VTP file");
        return false;
    }

    mVertices.resize(points.size() / 3);
    for (size_t i = 0; i < mVertices.size(); i++)
    {
        mVertices[i] = glm::vec3(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    }

    // Parse connectivity (indices)
    std::vector<float> connectivity = parseDataArray(content, "connectivity");
    if (connectivity.empty())
    {
        // Try alternative format
        std::vector<float> cells = parseDataArray(content, "Cells");
        if (!cells.empty())
        {
            mIndices.resize(cells.size());
            for (size_t i = 0; i < cells.size(); i++)
            {
                mIndices[i] = static_cast<uint32_t>(cells[i]);
            }
        }
        else
        {
            SPDLOG_ERROR("No connectivity data found in VTP file");
            return false;
        }
    }
    else
    {
        mIndices.resize(connectivity.size());
        for (size_t i = 0; i < connectivity.size(); i++)
        {
            mIndices[i] = static_cast<uint32_t>(connectivity[i]);
        }
    }

    // Parse normals if available
    std::vector<float> normals = parseDataArray(content, "Normals");
    if (!normals.empty() && normals.size() == mVertices.size() * 3)
    {
        mNormals.resize(mVertices.size());
        for (size_t i = 0; i < mVertices.size(); i++)
        {
            mNormals[i] = glm::vec3(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
        }
    }
    else
    {
        computeNormals();
    }

    computeBoundingBox();
    
    SPDLOG_INFO("Loaded mesh: {} vertices, {} triangles", mVertices.size(), mIndices.size() / 3);
    SPDLOG_INFO("Bounding box: [{}, {}, {}] - [{}, {}, {}]",
                mBBox.min.x, mBBox.min.y, mBBox.min.z,
                mBBox.max.x, mBBox.max.y, mBBox.max.z);

    return true;
}

std::vector<float> Mesh::parseDataArray(const std::string& content, const std::string& arrayName)
{
    std::vector<float> data;
    
    // Look for DataArray with the given name
    std::regex arrayRegex("<DataArray[^>]*Name=\"" + arrayName + "\"[^>]*>([^<]*)</DataArray>");
    std::smatch match;
    
    if (std::regex_search(content, match, arrayRegex))
    {
        std::string dataStr = match[1].str();
        std::istringstream iss(dataStr);
        float value;
        while (iss >> value)
        {
            data.push_back(value);
        }
    }
    
    return data;
}

bool Mesh::parseOBJ(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        SPDLOG_ERROR("Cannot open file: {}", filePath);
        return false;
    }

    mVertices.clear();
    mIndices.clear();
    mNormals.clear();

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            float x, y, z;
            iss >> x >> y >> z;
            mVertices.push_back(glm::vec3(x, y, z));
        }
        else if (prefix == "vn")
        {
            float x, y, z;
            iss >> x >> y >> z;
            mNormals.push_back(glm::vec3(x, y, z));
        }
        else if (prefix == "f")
        {
            std::string vertex1, vertex2, vertex3;
            iss >> vertex1 >> vertex2 >> vertex3;

            // Parse face indices (handle format: v/vt/vn)
            auto parseIndex = [](const std::string& str) -> uint32_t {
                size_t pos = str.find('/');
                std::string indexStr = (pos == std::string::npos) ? str : str.substr(0, pos);
                return static_cast<uint32_t>(std::stoi(indexStr)) - 1; // OBJ is 1-indexed
            };

            mIndices.push_back(parseIndex(vertex1));
            mIndices.push_back(parseIndex(vertex2));
            mIndices.push_back(parseIndex(vertex3));
        }
    }

    file.close();

    if (mVertices.empty() || mIndices.empty())
    {
        SPDLOG_ERROR("No valid mesh data found in OBJ file");
        return false;
    }

    // Compute normals if not provided
    if (mNormals.empty())
    {
        computeNormals();
    }

    computeBoundingBox();

    SPDLOG_INFO("Loaded OBJ mesh: {} vertices, {} triangles", mVertices.size(), mIndices.size() / 3);
    SPDLOG_INFO("Bounding box: [{}, {}, {}] - [{}, {}, {}]",
                mBBox.min.x, mBBox.min.y, mBBox.min.z,
                mBBox.max.x, mBBox.max.y, mBBox.max.z);

    return true;
}

void Mesh::computeBoundingBox()
{
    glm::vec3 min(INFINITY);
    glm::vec3 max(-INFINITY);
    
    for (const glm::vec3& vert : mVertices)
    {
        min.x = std::min(min.x, vert.x);
        max.x = std::max(max.x, vert.x);
        min.y = std::min(min.y, vert.y);
        max.y = std::max(max.y, vert.y);
        min.z = std::min(min.z, vert.z);
        max.z = std::max(max.z, vert.z);
    }
    
    mBBox = BoundingBox(min, max);
}

void Mesh::computeNormals()
{
    mNormals.clear();
    mNormals.assign(mVertices.size(), glm::vec3(0.0f));

    for (size_t i = 0; i < mIndices.size(); i += 3)
    {
        const glm::vec3 v1 = mVertices[mIndices[i]];
        const glm::vec3 v2 = mVertices[mIndices[i + 1]];
        const glm::vec3 v3 = mVertices[mIndices[i + 2]];
        
        const glm::vec3 normal = glm::normalize(glm::cross(v2 - v1, v3 - v1));

        // Weight by angle
        float angle1 = std::acos(glm::clamp(glm::dot(glm::normalize(v2 - v1), glm::normalize(v3 - v1)), -1.0f, 1.0f));
        float angle2 = std::acos(glm::clamp(glm::dot(glm::normalize(v1 - v2), glm::normalize(v3 - v2)), -1.0f, 1.0f));
        float angle3 = std::acos(glm::clamp(glm::dot(glm::normalize(v1 - v3), glm::normalize(v2 - v3)), -1.0f, 1.0f));

        mNormals[mIndices[i]] += angle1 * normal;
        mNormals[mIndices[i + 1]] += angle2 * normal;
        mNormals[mIndices[i + 2]] += angle3 * normal;
    }

    for (glm::vec3& n : mNormals)
    {
        n = glm::normalize(n);
    }
}

void Mesh::applyTransform(glm::mat4 trans)
{
    for (glm::vec3& vert : mVertices)
    {
        vert = glm::vec3(trans * glm::vec4(vert, 1.0f));
    }

    computeBoundingBox();
}

}
