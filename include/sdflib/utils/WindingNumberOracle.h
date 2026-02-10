#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace sdflib
{

class WindingNumberOracle
{
public:
    struct Settings
    {
        double theta = 0.25;
        uint32_t leafMaxTriangles = 8;
    };

    WindingNumberOracle(const std::vector<glm::vec3>& vertices,
                        const std::vector<uint32_t>& indices,
                        Settings settings = {});

    ~WindingNumberOracle();

    WindingNumberOracle(WindingNumberOracle&&) noexcept;
    WindingNumberOracle& operator=(WindingNumberOracle&&) noexcept;

    WindingNumberOracle(const WindingNumberOracle&) = delete;
    WindingNumberOracle& operator=(const WindingNumberOracle&) = delete;

    bool inside(const glm::vec3& p) const;
    double winding(const glm::vec3& p) const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace sdflib

