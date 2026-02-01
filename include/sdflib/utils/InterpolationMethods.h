#ifndef INTERPOLATION_METHODS_H
#define INTERPOLATION_METHODS_H

#include <array>

#include "utils/TriangleUtils.h"

#ifdef ENOKI_AVAILABLE
#include "enoki/array.h"
#endif

namespace sdflib
{
struct NoneInterpolation
{
    static constexpr uint32_t VALUES_PER_VERTEX = 0;
    static constexpr uint32_t EXTRA_VALUES = 0;
    static constexpr uint32_t NUM_COEFFICIENTS = 0;

    inline static void calculateCoefficients(const std::array<std::array<float, VALUES_PER_VERTEX>, 8>& valuesPerVertex,
                                      float nodeSize,
                                      const std::vector<uint32_t>& triangles,
                                      const Mesh& mesh,
                                      const std::vector<TriangleUtils::TriangleData>& trianglesData,
                                      std::array<float, NUM_COEFFICIENTS>& outCoefficients) {}

    inline static void calculatePointValues(glm::vec3 point,
                                      uint32_t nearestTriangleIndex,
                                      const Mesh& mesh,
                                      const std::vector<TriangleUtils::TriangleData>& trianglesData, 
                                      std::array<float, VALUES_PER_VERTEX>& outValues)
    { }

    inline static float interpolateValue(const std::array<float, NUM_COEFFICIENTS>& coefficients, glm::vec3 fracPart) 
    {
        return 0.0f;
    }

    inline static glm::vec3 interpolateGradient(const std::array<float, NUM_COEFFICIENTS>& values, glm::vec3 fracPart) 
    {
        return glm::vec3(0.0f);
    }

    inline static void interpolateVertexValues(const std::array<float, NUM_COEFFICIENTS>& values, glm::vec3 fracPart, float nodeSize, std::array<float, VALUES_PER_VERTEX>& outValues)
    {}
};

struct TriLinearInterpolation
{
    static constexpr uint32_t VALUES_PER_VERTEX = 1;
    static constexpr uint32_t EXTRA_VALUES = 0;
    static constexpr uint32_t NUM_COEFFICIENTS = 8;

    inline static void calculateCoefficients(const std::array<std::array<float, VALUES_PER_VERTEX>, 8>& valuesPerVertex,
                                             float nodeSize,
                                             const std::vector<uint32_t>& triangles,
                                             const Mesh& mesh,
                                             const std::vector<TriangleUtils::TriangleData>& trianglesData,
                                             std::array<float, NUM_COEFFICIENTS>& outCoefficients) 
    {
        outCoefficients = *reinterpret_cast<const std::array<float, NUM_COEFFICIENTS>*>(&valuesPerVertex);
    }

    inline static void calculatePointValues(glm::vec3 point,
                                      uint32_t nearestTriangleIndex,
                                      const Mesh& mesh,
                                      const std::vector<TriangleUtils::TriangleData>& trianglesData, 
                                      std::array<float, VALUES_PER_VERTEX>& outValues)
    { 
        outValues[0] = TriangleUtils::getSignedDistPointAndTriangle(point, trianglesData[nearestTriangleIndex]);
    }

    inline static float interpolateValue(const std::array<float, NUM_COEFFICIENTS>& values, glm::vec3 fracPart) 
    {
        float d00 = values[0] * (1.0f - fracPart.x) +
                values[1] * fracPart.x;
        float d01 = values[2] * (1.0f - fracPart.x) +
                    values[3] * fracPart.x;
        float d10 = values[4] * (1.0f - fracPart.x) +
                    values[5] * fracPart.x;
        float d11 = values[6] * (1.0f - fracPart.x) +
                    values[7] * fracPart.x;

        float d0 = d00 * (1.0f - fracPart.y) + d01 * fracPart.y;
        float d1 = d10 * (1.0f - fracPart.y) + d11 * fracPart.y;

        return d0 * (1.0f - fracPart.z) + d1 * fracPart.z;
    }

    inline static glm::vec3 interpolateGradient(const std::array<float, NUM_COEFFICIENTS>& values, glm::vec3 fracPart) 
    {
        float gx;
        {
            float d00 = values[0] * (1.0f - fracPart.y) +
                values[2] * fracPart.y;
            float d01 = values[1] * (1.0f - fracPart.y) +
                        values[3] * fracPart.y;
            float d10 = values[4] * (1.0f - fracPart.y) +
                        values[6] * fracPart.y;
            float d11 = values[5] * (1.0f - fracPart.y) +
                        values[7] * fracPart.y;

            float d0 = d00 * (1.0f - fracPart.z) + d10 * fracPart.z;
            float d1 = d01 * (1.0f - fracPart.z) + d11 * fracPart.z;

            gx = d1 - d0;
        }

        float gy;
        float gz;
        {
            float d00 = values[0] * (1.0f - fracPart.x) +
                values[1] * fracPart.x;
            float d01 = values[2] * (1.0f - fracPart.x) +
                        values[3] * fracPart.x;
            float d10 = values[4] * (1.0f - fracPart.x) +
                        values[5] * fracPart.x;
            float d11 = values[6] * (1.0f - fracPart.x) +
                        values[7] * fracPart.x;

            {
                float d0 = d00 * (1.0f - fracPart.z) + d10 * fracPart.z;
                float d1 = d01 * (1.0f - fracPart.z) + d11 * fracPart.z;

                gy = d1 - d0;
            }

            {
                float d0 = d00 * (1.0f - fracPart.y) + d01 * fracPart.y;
                float d1 = d10 * (1.0f - fracPart.y) + d11 * fracPart.y;

                gz = d1 - d0;
            }
        }

        return glm::vec3(gx, gy, gz);
    }

    inline static void interpolateVertexValues(const std::array<float, NUM_COEFFICIENTS>& values, glm::vec3 fracPart, float nodeSize, std::array<float, VALUES_PER_VERTEX>& outValues)
    {
        outValues[0] = interpolateValue(values, fracPart);
    }
};
}

#endif
