/**
 * @file BoundingBox.h
 * @brief Axis-aligned bounding box implementation
 */

#pragma once

#include <glm/glm.hpp>
#include <array>
#include <algorithm>
#include <cereal/cereal.hpp>

// glm::vec3 serialization for cereal
namespace glm
{
    template<class Archive>
    void serialize(Archive& archive, glm::vec3& v)
    {
        archive(v.x, v.y, v.z);
    }
}

namespace sdflib
{

class BoundingBox
{
public:
    glm::vec3 min;
    glm::vec3 max;

    BoundingBox() : min(glm::vec3(0.0f)), max(glm::vec3(0.0f)) {}
    BoundingBox(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    glm::vec3 getCenter() const { return (min + max) * 0.5f; }
    glm::vec3 getSize() const { return max - min; }
    
    float getDistance(const glm::vec3& point) const
    {
        glm::vec3 closest = glm::clamp(point, min, max);
        return glm::length(point - closest);
    }

    float getDistance(const glm::vec3& point, glm::vec3& outGradient) const
    {
        glm::vec3 closest = glm::clamp(point, min, max);
        outGradient = glm::normalize(point - closest);
        return glm::length(point - closest);
    }

    void addMargin(float margin)
    {
        min -= glm::vec3(margin);
        max += glm::vec3(margin);
    }

    bool contains(const glm::vec3& point) const
    {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }

    template<class Archive>
    void serialize(Archive& archive)
    {
        archive(min, max);
    }
};

}
