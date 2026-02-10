#include "sdflib/utils/WindingNumberOracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace sdflib
{

static inline glm::dvec3 toD(const glm::vec3& v)
{
    return glm::dvec3(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
}

struct WindingNumberOracle::Impl
{
    struct Aabb
    {
        glm::dvec3 min = glm::dvec3(std::numeric_limits<double>::infinity());
        glm::dvec3 max = glm::dvec3(-std::numeric_limits<double>::infinity());
    };

    struct Node
    {
        Aabb box;
        glm::dvec3 centroid = glm::dvec3(0.0);
        glm::dvec3 areaVec = glm::dvec3(0.0);
        double radius = 0.0;
        uint32_t start = 0;
        uint32_t count = 0;
        uint32_t left = std::numeric_limits<uint32_t>::max();
        uint32_t right = std::numeric_limits<uint32_t>::max();
        bool leaf = false;
    };

    Settings settings{};
    const std::vector<glm::vec3>* vertices = nullptr;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<glm::dvec3> faceCentroids;
    std::vector<glm::dvec3> faceAreaVec;
    std::vector<double> faceAbsArea;
    std::vector<uint32_t> triOrder;
    std::vector<Node> nodes;
    uint32_t root = std::numeric_limits<uint32_t>::max();

    static inline uint64_t edgeKey(uint32_t a, uint32_t b)
    {
        uint32_t lo = a < b ? a : b;
        uint32_t hi = a < b ? b : a;
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    }

    static inline int edgeDir(const std::array<uint32_t, 3>& f, uint32_t a, uint32_t b)
    {
        if (f[0] == a && f[1] == b) return 1;
        if (f[1] == a && f[2] == b) return 1;
        if (f[2] == a && f[0] == b) return 1;
        if (f[0] == b && f[1] == a) return -1;
        if (f[1] == b && f[2] == a) return -1;
        if (f[2] == b && f[0] == a) return -1;
        return 0;
    }

    static inline void expandAabb(Aabb& b, const glm::dvec3& p)
    {
        b.min = glm::min(b.min, p);
        b.max = glm::max(b.max, p);
    }

    static inline double solidAngleTriangle(const glm::dvec3& p, const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c)
    {
        const glm::dvec3 ra = a - p;
        const glm::dvec3 rb = b - p;
        const glm::dvec3 rc = c - p;

        const double la = glm::length(ra);
        const double lb = glm::length(rb);
        const double lc = glm::length(rc);

        const double num = glm::dot(ra, glm::cross(rb, rc));
        const double den = la * lb * lc +
                           glm::dot(ra, rb) * lc +
                           glm::dot(rb, rc) * la +
                           glm::dot(rc, ra) * lb;

        if (!std::isfinite(num) || !std::isfinite(den)) return 0.0;

        return 2.0 * std::atan2(num, den);
    }

    void orientFaces()
    {
        std::unordered_map<uint64_t, std::vector<uint32_t>> edgeToFaces;
        edgeToFaces.reserve(faces.size() * 3);

        for (uint32_t f = 0; f < static_cast<uint32_t>(faces.size()); f++)
        {
            const auto& tri = faces[f];
            edgeToFaces[edgeKey(tri[0], tri[1])].push_back(f);
            edgeToFaces[edgeKey(tri[1], tri[2])].push_back(f);
            edgeToFaces[edgeKey(tri[2], tri[0])].push_back(f);
        }

        std::vector<uint8_t> visited(faces.size(), 0);
        std::queue<uint32_t> q;

        for (uint32_t seed = 0; seed < static_cast<uint32_t>(faces.size()); seed++)
        {
            if (visited[seed]) continue;
            visited[seed] = 1;
            q.push(seed);

            while (!q.empty())
            {
                const uint32_t f = q.front();
                q.pop();

                const auto tri = faces[f];
                const std::array<std::pair<uint32_t, uint32_t>, 3> edges = {
                    std::make_pair(tri[0], tri[1]),
                    std::make_pair(tri[1], tri[2]),
                    std::make_pair(tri[2], tri[0])};

                for (const auto& e : edges)
                {
                    const uint64_t k = edgeKey(e.first, e.second);
                    auto it = edgeToFaces.find(k);
                    if (it == edgeToFaces.end()) continue;
                    const auto& adj = it->second;
                    if (adj.size() < 2) continue;

                    for (uint32_t g : adj)
                    {
                        if (g == f) continue;
                        if (visited[g]) continue;

                        const int df = edgeDir(faces[f], e.first, e.second);
                        const int dg = edgeDir(faces[g], e.first, e.second);
                        if (df != 0 && dg != 0 && df == dg)
                        {
                            std::swap(faces[g][1], faces[g][2]);
                        }

                        visited[g] = 1;
                        q.push(g);
                    }
                }
            }
        }
    }

    void buildFaceData()
    {
        faceCentroids.resize(faces.size());
        faceAreaVec.resize(faces.size());
        faceAbsArea.resize(faces.size());

        for (size_t i = 0; i < faces.size(); i++)
        {
            const auto& f = faces[i];
            const glm::dvec3 a = toD((*vertices)[f[0]]);
            const glm::dvec3 b = toD((*vertices)[f[1]]);
            const glm::dvec3 c = toD((*vertices)[f[2]]);
            faceCentroids[i] = (a + b + c) / 3.0;
            const glm::dvec3 av = 0.5 * glm::cross(b - a, c - a);
            faceAreaVec[i] = av;
            faceAbsArea[i] = glm::length(av);
        }
    }

    Node buildNode(uint32_t start, uint32_t count)
    {
        Node n;
        n.start = start;
        n.count = count;

        double areaSum = 0.0;
        glm::dvec3 centroidSum(0.0);
        glm::dvec3 areaVecSum(0.0);

        Aabb centroidBox;

        for (uint32_t i = 0; i < count; i++)
        {
            const uint32_t triIdx = triOrder[start + i];
            const auto& f = faces[triIdx];
            const glm::dvec3 a = toD((*vertices)[f[0]]);
            const glm::dvec3 b = toD((*vertices)[f[1]]);
            const glm::dvec3 c = toD((*vertices)[f[2]]);
            expandAabb(n.box, a);
            expandAabb(n.box, b);
            expandAabb(n.box, c);

            const glm::dvec3 cen = faceCentroids[triIdx];
            expandAabb(centroidBox, cen);

            const double w = faceAbsArea[triIdx];
            areaSum += w;
            centroidSum += w * cen;
            areaVecSum += faceAreaVec[triIdx];
        }

        n.centroid = (areaSum > 0.0) ? (centroidSum / areaSum) : ((n.box.min + n.box.max) * 0.5);
        n.areaVec = areaVecSum;

        const glm::dvec3 halfDiag = (n.box.max - n.box.min) * 0.5;
        n.radius = glm::length(halfDiag);

        if (count <= settings.leafMaxTriangles)
        {
            n.leaf = true;
            return n;
        }

        const glm::dvec3 cSize = centroidBox.max - centroidBox.min;
        int axis = 0;
        if (cSize.y > cSize.x) axis = 1;
        if (cSize.z > cSize[axis]) axis = 2;

        const uint32_t mid = start + count / 2;
        auto begin = triOrder.begin() + start;
        auto middle = triOrder.begin() + mid;
        auto end = triOrder.begin() + (start + count);

        std::nth_element(begin, middle, end, [&](uint32_t a, uint32_t b) {
            return faceCentroids[a][axis] < faceCentroids[b][axis];
        });

        return n;
    }

    uint32_t buildBvh(uint32_t start, uint32_t count)
    {
        Node n = buildNode(start, count);
        const uint32_t nodeIdx = static_cast<uint32_t>(nodes.size());
        nodes.push_back(n);

        if (n.leaf) return nodeIdx;

        const uint32_t leftCount = count / 2;
        const uint32_t rightCount = count - leftCount;

        if (leftCount == 0 || rightCount == 0)
        {
            nodes[nodeIdx].leaf = true;
            return nodeIdx;
        }

        const uint32_t left = buildBvh(start, leftCount);
        const uint32_t right = buildBvh(start + leftCount, rightCount);

        nodes[nodeIdx].left = left;
        nodes[nodeIdx].right = right;

        return nodeIdx;
    }

    double approxSolidAngleNode(const Node& n, const glm::dvec3& p) const
    {
        const glm::dvec3 r = n.centroid - p;
        const double d2 = glm::dot(r, r);
        if (d2 <= 0.0) return 0.0;
        const double inv = 1.0 / (d2 * std::sqrt(d2));
        return glm::dot(n.areaVec, r) * inv;
    }

    void accumulateOmega(uint32_t nodeIdx, const glm::dvec3& p, double theta2, double& outOmega) const
    {
        const Node& n = nodes[nodeIdx];
        const glm::dvec3 r = n.centroid - p;
        const double d2 = glm::dot(r, r);

        if (!n.leaf && d2 > 0.0)
        {
            const double ratio2 = (n.radius * n.radius) / d2;
            if (ratio2 < theta2)
            {
                outOmega += approxSolidAngleNode(n, p);
                return;
            }
        }

        if (n.leaf)
        {
            for (uint32_t i = 0; i < n.count; i++)
            {
                const uint32_t triIdx = triOrder[n.start + i];
                const auto& f = faces[triIdx];
                const glm::dvec3 a = toD((*vertices)[f[0]]);
                const glm::dvec3 b = toD((*vertices)[f[1]]);
                const glm::dvec3 c = toD((*vertices)[f[2]]);
                outOmega += solidAngleTriangle(p, a, b, c);
            }
            return;
        }

        if (n.left != std::numeric_limits<uint32_t>::max()) accumulateOmega(n.left, p, theta2, outOmega);
        if (n.right != std::numeric_limits<uint32_t>::max()) accumulateOmega(n.right, p, theta2, outOmega);
    }
};

WindingNumberOracle::WindingNumberOracle(const std::vector<glm::vec3>& vertices,
                                         const std::vector<uint32_t>& indices,
                                         Settings settings)
    : mImpl(std::make_unique<Impl>())
{
    if (indices.size() % 3 != 0)
    {
        throw std::runtime_error("WindingNumberOracle: indices size must be multiple of 3");
    }

    mImpl->settings = settings;
    mImpl->vertices = &vertices;

    const size_t triCount = indices.size() / 3;
    mImpl->faces.resize(triCount);
    for (size_t t = 0; t < triCount; t++)
    {
        mImpl->faces[t] = {indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]};
    }

    mImpl->orientFaces();
    mImpl->buildFaceData();

    mImpl->triOrder.resize(triCount);
    for (uint32_t i = 0; i < static_cast<uint32_t>(triCount); i++) mImpl->triOrder[i] = i;

    mImpl->nodes.reserve(triCount * 2);
    mImpl->root = mImpl->buildBvh(0u, static_cast<uint32_t>(triCount));
}

WindingNumberOracle::~WindingNumberOracle() = default;

WindingNumberOracle::WindingNumberOracle(WindingNumberOracle&&) noexcept = default;
WindingNumberOracle& WindingNumberOracle::operator=(WindingNumberOracle&&) noexcept = default;

double WindingNumberOracle::winding(const glm::vec3& p) const
{
    if (!mImpl || mImpl->root == std::numeric_limits<uint32_t>::max()) return 0.0;

    const glm::dvec3 pd = toD(p);
    double omega = 0.0;
    const double theta2 = mImpl->settings.theta * mImpl->settings.theta;
    mImpl->accumulateOmega(mImpl->root, pd, theta2, omega);
    static constexpr double inv4Pi = 0.079577471545947667884441881686257181017229822870228;
    return omega * inv4Pi;
}

bool WindingNumberOracle::inside(const glm::vec3& p) const
{
    const double w = winding(p);
    return std::abs(w) > 0.5;
}

} // namespace sdflib

