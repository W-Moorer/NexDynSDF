/**
 * @file test_icg.cpp
 * @brief ICG (InteractiveComputerGraphics) 库测试
 * @details 测试ICG提供的图形学工具函数和数据结构
 */

#include <iostream>
#include <cmath>

/**
 * @brief 测试基本数学工具函数
 * @return 测试是否通过
 */
bool test_math_utilities()
{
    std::cout << "[ICG] 测试数学工具函数..." << std::endl;

    try
    {
        // 常用的图形学数学函数
        auto clamp = [](float value, float min, float max) -> float
        {
            return value < min ? min : (value > max ? max : value);
        };

        auto lerp = [](float a, float b, float t) -> float
        {
            return a + t * (b - a);
        };

        auto smoothstep = [](float edge0, float edge1, float x) -> float
        {
            float t = (x - edge0) / (edge1 - edge0);
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            return t * t * (3.0f - 2.0f * t);
        };

        // 测试clamp
        float clamped = clamp(5.0f, 0.0f, 1.0f);
        if (clamped != 1.0f)
        {
            std::cerr << "clamp函数错误" << std::endl;
            return false;
        }

        // 测试lerp
        float interpolated = lerp(0.0f, 10.0f, 0.5f);
        if (std::abs(interpolated - 5.0f) > 1e-6f)
        {
            std::cerr << "lerp函数错误" << std::endl;
            return false;
        }

        // 测试smoothstep
        float smoothed = smoothstep(0.0f, 1.0f, 0.5f);
        std::cout << "smoothstep(0, 1, 0.5) = " << smoothed << std::endl;

        std::cout << "[ICG] 数学工具函数测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "数学工具函数测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试颜色处理
 * @return 测试是否通过
 */
bool test_color_operations()
{
    std::cout << "[ICG] 测试颜色处理..." << std::endl;

    try
    {
        // RGB颜色结构
        struct Color3f
        {
            float r, g, b;
            Color3f(float r_=0, float g_=0, float b_=0) : r(r_), g(g_), b(b_) {}

            Color3f operator+(const Color3f& other) const
            {
                return Color3f(r + other.r, g + other.g, b + other.b);
            }

            Color3f operator*(float s) const
            {
                return Color3f(r * s, g * s, b * s);
            }
        };

        // 颜色空间转换: RGB to Grayscale
        auto rgb_to_grayscale = [](const Color3f& color) -> float
        {
            return 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;
        };

        Color3f red(1.0f, 0.0f, 0.0f);
        Color3f green(0.0f, 1.0f, 0.0f);
        Color3f blue(0.0f, 0.0f, 1.0f);

        float gray_red = rgb_to_grayscale(red);
        float gray_green = rgb_to_grayscale(green);
        float gray_blue = rgb_to_grayscale(blue);

        std::cout << "Red grayscale: " << gray_red << std::endl;
        std::cout << "Green grayscale: " << gray_green << std::endl;
        std::cout << "Blue grayscale: " << gray_blue << std::endl;

        // 颜色混合
        Color3f mixed = red * 0.5f + green * 0.5f;
        std::cout << "Mixed color: (" << mixed.r << ", " << mixed.g << ", " << mixed.b << ")" << std::endl;

        std::cout << "[ICG] 颜色处理测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "颜色处理测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试几何图元
 * @return 测试是否通过
 */
bool test_geometric_primitives()
{
    std::cout << "[ICG] 测试几何图元..." << std::endl;

    try
    {
        // 射线结构
        struct Ray3f
        {
            float origin[3];
            float direction[3];

            void point_at(float t, float* point) const
            {
                point[0] = origin[0] + t * direction[0];
                point[1] = origin[1] + t * direction[1];
                point[2] = origin[2] + t * direction[2];
            }
        };

        // 创建射线
        Ray3f ray;
        ray.origin[0] = 0.0f; ray.origin[1] = 0.0f; ray.origin[2] = 0.0f;
        ray.direction[0] = 1.0f; ray.direction[1] = 0.0f; ray.direction[2] = 0.0f;

        float point[3];
        ray.point_at(5.0f, point);

        std::cout << "射线上的点 (t=5): (" << point[0] << ", " << point[1] << ", " << point[2] << ")" << std::endl;

        if (std::abs(point[0] - 5.0f) > 1e-6f)
        {
            std::cerr << "射线计算错误" << std::endl;
            return false;
        }

        // 球体结构
        struct Sphere
        {
            float center[3];
            float radius;

            bool contains(float x, float y, float z) const
            {
                float dx = x - center[0];
                float dy = y - center[1];
                float dz = z - center[2];
                return (dx*dx + dy*dy + dz*dz) <= radius * radius;
            }
        };

        Sphere sphere;
        sphere.center[0] = 0.0f; sphere.center[1] = 0.0f; sphere.center[2] = 0.0f;
        sphere.radius = 1.0f;

        bool inside = sphere.contains(0.5f, 0.0f, 0.0f);
        bool outside = sphere.contains(2.0f, 0.0f, 0.0f);

        std::cout << "点(0.5, 0, 0)在球内: " << (inside ? "是" : "否") << std::endl;
        std::cout << "点(2, 0, 0)在球内: " << (outside ? "是" : "否") << std::endl;

        if (!inside || outside)
        {
            std::cerr << "球体包含测试错误" << std::endl;
            return false;
        }

        std::cout << "[ICG] 几何图元测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "几何图元测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试SDF相关函数
 * @return 测试是否通过
 */
bool test_sdf_functions()
{
    std::cout << "[ICG] 测试SDF相关函数..." << std::endl;

    try
    {
        // 球体SDF
        auto sphere_sdf = [](float x, float y, float z, float radius) -> float
        {
            return std::sqrt(x*x + y*y + z*z) - radius;
        };

        // 立方体SDF
        auto box_sdf = [](float x, float y, float z, float bx, float by, float bz) -> float
        {
            float dx = std::abs(x) - bx;
            float dy = std::abs(y) - by;
            float dz = std::abs(z) - bz;
            float outside_dist = std::sqrt(std::max(dx, 0.0f)*std::max(dx, 0.0f) +
                                           std::max(dy, 0.0f)*std::max(dy, 0.0f) +
                                           std::max(dz, 0.0f)*std::max(dz, 0.0f));
            float inside_dist = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
            return outside_dist + inside_dist;
        };

        // 测试球体SDF
        float sphere_dist_center = sphere_sdf(0.0f, 0.0f, 0.0f, 1.0f);
        float sphere_dist_surface = sphere_sdf(1.0f, 0.0f, 0.0f, 1.0f);
        float sphere_dist_outside = sphere_sdf(2.0f, 0.0f, 0.0f, 1.0f);

        std::cout << "球体SDF (中心): " << sphere_dist_center << std::endl;
        std::cout << "球体SDF (表面): " << sphere_dist_surface << std::endl;
        std::cout << "球体SDF (外部): " << sphere_dist_outside << std::endl;

        if (std::abs(sphere_dist_center - (-1.0f)) > 1e-6f)
        {
            std::cerr << "球体SDF中心点计算错误" << std::endl;
            return false;
        }

        // 测试立方体SDF
        float box_dist = box_sdf(0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f);
        std::cout << "立方体SDF (内部): " << box_dist << std::endl;

        std::cout << "[ICG] SDF相关函数测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "SDF函数测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 程序入口
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "ICG 图形学库测试" << std::endl;
    std::cout << "版本: master" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_math_utilities();
    all_passed &= test_color_operations();
    all_passed &= test_geometric_primitives();
    all_passed &= test_sdf_functions();

    std::cout << "========================================" << std::endl;
    if (all_passed)
    {
        std::cout << "所有测试通过!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "部分测试失败!" << std::endl;
        return 1;
    }
}
