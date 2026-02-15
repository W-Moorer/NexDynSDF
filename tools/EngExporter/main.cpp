#include "sdflib/utils/MeshBinaryLoader.h"
#include "sdflib/utils/NagataEnhanced.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <map>

int main(int argc, char** argv)
{
    if (argc == 4 && std::string(argv[1]) == "--compare")
    {
        std::map<sdflib::NagataEnhanced::EdgeKey, glm::dvec3> baseData;
        std::map<sdflib::NagataEnhanced::EdgeKey, glm::dvec3> cmpData;

        if (!sdflib::NagataEnhanced::loadEnhancedDataDouble(argv[2], baseData))
        {
            std::cerr << "Failed to load baseline ENG data" << std::endl;
            return 1;
        }

        if (!sdflib::NagataEnhanced::loadEnhancedDataDouble(argv[3], cmpData))
        {
            std::cerr << "Failed to load compare ENG data" << std::endl;
            return 1;
        }

        std::vector<sdflib::NagataEnhanced::EdgeKey> onlyBase;
        std::vector<sdflib::NagataEnhanced::EdgeKey> onlyCmp;
        struct DiffEntry
        {
            sdflib::NagataEnhanced::EdgeKey key;
            glm::dvec3 base;
            glm::dvec3 cmp;
        };
        std::vector<DiffEntry> mismatched;
        double maxAbsComp = 0.0;
        double maxL2 = 0.0;
        double sumL2 = 0.0;
        double sumL2Sq = 0.0;
        size_t matchedCount = 0;

        for (const auto& [key, baseVal] : baseData)
        {
            auto it = cmpData.find(key);
            if (it == cmpData.end())
            {
                onlyBase.push_back(key);
                continue;
            }
            glm::dvec3 diff = it->second - baseVal;
            double absComp = std::max({std::abs(diff.x), std::abs(diff.y), std::abs(diff.z)});
            double l2 = glm::length(diff);
            maxAbsComp = std::max(maxAbsComp, absComp);
            maxL2 = std::max(maxL2, l2);
            sumL2 += l2;
            sumL2Sq += l2 * l2;
            matchedCount += 1;
            if (l2 > 0.0)
            {
                mismatched.push_back({key, baseVal, it->second});
            }
        }

        for (const auto& [key, cmpVal] : cmpData)
        {
            if (baseData.find(key) == baseData.end())
            {
                onlyCmp.push_back(key);
            }
        }

        std::cout << "baseline edges: " << baseData.size() << std::endl;
        std::cout << "compare edges: " << cmpData.size() << std::endl;
        std::cout << "only baseline edges: " << onlyBase.size() << std::endl;
        std::cout << "only compare edges: " << onlyCmp.size() << std::endl;
        std::cout << "coeff mismatches: " << mismatched.size() << std::endl;
        
        if (matchedCount > 0)
        {
            double meanL2 = sumL2 / static_cast<double>(matchedCount);
            double rmsL2 = std::sqrt(sumL2Sq / static_cast<double>(matchedCount));
            std::cout << std::setprecision(12);
            std::cout << "max_abs_component: " << maxAbsComp << std::endl;
            std::cout << "max_l2: " << maxL2 << std::endl;
            std::cout << "mean_l2: " << meanL2 << std::endl;
            std::cout << "rms_l2: " << rmsL2 << std::endl;
        }

        size_t maxShow = 10;
        for (size_t i = 0; i < mismatched.size() && i < maxShow; ++i)
        {
            const auto& d = mismatched[i];
            std::cout << "diff edge (" << d.key.v0 << "," << d.key.v1 << ") "
                      << "base " << d.base.x << "," << d.base.y << "," << d.base.z << " "
                      << "cmp " << d.cmp.x << "," << d.cmp.y << "," << d.cmp.z << std::endl;
        }

        for (size_t i = 0; i < onlyBase.size() && i < maxShow; ++i)
        {
            const auto& k = onlyBase[i];
            const auto& v = baseData.find(k)->second;
            std::cout << "only baseline edge (" << k.v0 << "," << k.v1 << ") "
                      << v.x << "," << v.y << "," << v.z << std::endl;
        }

        for (size_t i = 0; i < onlyCmp.size() && i < maxShow; ++i)
        {
            const auto& k = onlyCmp[i];
            const auto& v = cmpData.find(k)->second;
            std::cout << "only compare edge (" << k.v0 << "," << k.v1 << ") "
                      << v.x << "," << v.y << "," << v.z << std::endl;
        }

        return 0;
    }

    if (argc < 3)
    {
        std::cerr << "Usage: EngExporter <input.nsm> <output.eng>" << std::endl;
        std::cerr << "   or: EngExporter --compare <baseline.eng> <compare.eng>" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    auto meshData = sdflib::MeshBinaryLoader::loadFromNSMDouble(inputPath);
    if (meshData.vertices.empty())
    {
        std::cerr << "Failed to load NSM data" << std::endl;
        return 1;
    }

    auto creaseEdges = sdflib::NagataEnhanced::detectCreaseEdgesD(
        meshData.vertices, meshData.faces, meshData.faceNormals);

    auto cSharpsD = sdflib::NagataEnhanced::computeCSharpForEdgesD(creaseEdges);
    
    if (!sdflib::NagataEnhanced::saveEnhancedDataDouble(outputPath, cSharpsD))
    {
        std::cerr << "Failed to save ENG data" << std::endl;
        return 1;
    }

    std::cout << "Saved ENG data to " << outputPath << std::endl;
    return 0;
}
