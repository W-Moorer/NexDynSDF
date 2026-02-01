/**
 * @file SdfFunction.cpp
 * @brief Base SDF class with serialization support
 */

#include "sdflib/SdfFunction.h"
#include "sdflib/OctreeSdf.h"
#include "sdflib/ExactOctreeSdf.h"
#include <spdlog/spdlog.h>
#include <fstream>

namespace sdflib
{

bool SdfFunction::saveToFile(const std::string& outputPath)
{
    std::ofstream os(outputPath, std::ios::out | std::ios::binary);
    if (!os.is_open())
    {
        SPDLOG_ERROR("Cannot open file {}", outputPath);
        return false;
    }

    cereal::PortableBinaryOutputArchive archive(os);
    SdfFormat format = getFormat();
    SPDLOG_INFO("Saving SDF with format: {}", static_cast<int>(format));

    if (format == SdfFormat::OCTREE)
    {
        archive(format);
        archive(*reinterpret_cast<OctreeSdf*>(this));
    }
    else if (format == SdfFormat::EXACT_OCTREE)
    {
        archive(format);
        archive(*reinterpret_cast<ExactOctreeSdf*>(this));
    }
    else
    {
        SPDLOG_ERROR("Unknown format to save");
        return false;
    }

    return true;
}

std::unique_ptr<SdfFunction> SdfFunction::loadFromFile(const std::string& inputPath)
{
    std::ifstream is(inputPath, std::ios::binary);
    if (!is.is_open())
    {
        SPDLOG_ERROR("Cannot open file {}", inputPath);
        return std::unique_ptr<SdfFunction>();
    }

    cereal::PortableBinaryInputArchive archive(is);
    SdfFunction::SdfFormat format = SdfFunction::SdfFormat::NONE;
    archive(format);

    if (format == SdfFormat::OCTREE)
    {
        std::unique_ptr<OctreeSdf> obj(new OctreeSdf());
        archive(*obj);
        return obj;
    }
    else if (format == SdfFormat::EXACT_OCTREE)
    {
        std::unique_ptr<ExactOctreeSdf> obj(new ExactOctreeSdf());
        archive(*obj);
        return obj;
    }
    else
    {
        SPDLOG_ERROR("Unknown file format");
        return std::unique_ptr<SdfFunction>();
    }
}

}
