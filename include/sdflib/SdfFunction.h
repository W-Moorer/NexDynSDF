/**
 * @file SdfFunction.h
 * @brief Base class for SDF implementations with serialization support
 */

#pragma once

#include "utils/BoundingBox.h"
#include <glm/glm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <string>

namespace sdflib
{

class SdfFunction
{
public:
    enum class SdfFormat
    {
        NONE = 0,
        GRID = 1,
        OCTREE = 2,
        EXACT_OCTREE = 3
    };

    virtual ~SdfFunction() = default;

    // Get signed distance at sample point
    virtual float getDistance(glm::vec3 sample) const = 0;
    
    // Get distance and gradient
    virtual float getDistance(glm::vec3 sample, glm::vec3& outGradient) const = 0;
    
    // Get bounding box
    virtual BoundingBox getBoundingBox() const = 0;
    
    // Get SDF format
    virtual SdfFormat getFormat() const = 0;

    // Save to file
    bool saveToFile(const std::string& outputPath);
    
    // Load from file
    static std::unique_ptr<SdfFunction> loadFromFile(const std::string& inputPath);
};

}
