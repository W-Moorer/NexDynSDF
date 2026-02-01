/**
 * @file test_enoki.cpp
 * @brief Enoki向量化优化库测试
 * @details 测试SIMD向量化、数组运算等功能
 */

#include <iostream>
#include <enoki/array.h>
#include <enoki/matrix.h>

using namespace enoki;

/**
 * @brief 测试基本SIMD数组运算
 * @return 测试是否通过
 */
bool test_simd_array_operations()
{
    std::cout << "[Enoki] 测试SIMD数组运算..." << std::endl;

    try
    {
        // 创建SIMD浮点数组
        Array<float, 4> a(1.0f, 2.0f, 3.0f, 4.0f);
        Array<float, 4> b(5.0f, 6.0f, 7.0f, 8.0f);

        // 基本运算
        Array<float, 4> c = a + b;
        Array<float, 4> d = a * b;

        std::cout << "a = " << a << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "a + b = " << c << std::endl;
        std::cout << "a * b = " << d << std::endl;

        // 水平求和
        float sum_a = hsum(a);
        std::cout << "sum(a) = " << sum_a << std::endl;

        if (std::abs(sum_a - 10.0f) > 1e-6f)
        {
            std::cerr << "水平求和结果错误" << std::endl;
            return false;
        }

        std::cout << "[Enoki] SIMD数组运算测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "SIMD数组运算测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试向量化数学函数
 * @return 测试是否通过
 */
bool test_vectorized_math()
{
    std::cout << "[Enoki] 测试向量化数学函数..." << std::endl;

    try
    {
        Array<float, 4> x(0.0f, 0.5f, 1.0f, 1.5f);

        // 向量化sin
        Array<float, 4> sin_x = sin(x);
        std::cout << "sin(" << x << ") = " << sin_x << std::endl;

        // 向量化cos
        Array<float, 4> cos_x = cos(x);
        std::cout << "cos(" << x << ") = " << cos_x << std::endl;

        // 向量化sqrt
        Array<float, 4> y(1.0f, 4.0f, 9.0f, 16.0f);
        Array<float, 4> sqrt_y = sqrt(y);
        std::cout << "sqrt(" << y << ") = " << sqrt_y << std::endl;

        // 验证结果
        if (std::abs(sqrt_y[0] - 1.0f) > 1e-5f ||
            std::abs(sqrt_y[1] - 2.0f) > 1e-5f ||
            std::abs(sqrt_y[2] - 3.0f) > 1e-5f ||
            std::abs(sqrt_y[3] - 4.0f) > 1e-5f)
        {
            std::cerr << "sqrt计算结果错误" << std::endl;
            return false;
        }

        std::cout << "[Enoki] 向量化数学函数测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "向量化数学函数测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试动态数组Packet
 * @return 测试是否通过
 */
bool test_dynamic_packets()
{
    std::cout << "[Enoki] 测试动态数组Packet..." << std::endl;

    try
    {
        // 使用Packet进行动态向量化
        Packet<float> p1(1.0f);
        Packet<float> p2(2.0f);

        Packet<float> p3 = p1 + p2;
        Packet<float> p4 = p1 * p2;

        std::cout << "Packet运算完成" << std::endl;

        std::cout << "[Enoki] 动态数组Packet测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "动态数组Packet测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试掩码操作
 * @return 测试是否通过
 */
bool test_mask_operations()
{
    std::cout << "[Enoki] 测试掩码操作..." << std::endl;

    try
    {
        Array<float, 4> a(1.0f, 2.0f, 3.0f, 4.0f);
        Array<float, 4> b(3.0f, 3.0f, 3.0f, 3.0f);

        // 创建掩码
        auto mask = a < b;
        std::cout << "掩码 (a < b): " << mask << std::endl;

        // 使用掩码进行选择
        Array<float, 4> result = select(mask, a, b);
        std::cout << "select(mask, a, b) = " << result << std::endl;

        // 计数
        size_t count = count_true(mask);
        std::cout << "满足条件的元素数: " << count << std::endl;

        std::cout << "[Enoki] 掩码操作测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "掩码操作测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 程序入口
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "Enoki 向量化优化库测试" << std::endl;
    std::cout << "版本: master" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_simd_array_operations();
    all_passed &= test_vectorized_math();
    all_passed &= test_dynamic_packets();
    all_passed &= test_mask_operations();

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
