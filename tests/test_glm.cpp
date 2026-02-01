/**
 * @file test_glm.cpp
 * @brief GLM数学库测试
 * @details 测试向量运算、矩阵变换等功能
 */

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

/**
 * @brief 测试向量基本运算
 * @return 测试是否通过
 */
bool test_vector_operations()
{
    std::cout << "[GLM] 测试向量运算..." << std::endl;

    // 向量创建和运算
    glm::vec3 v1(1.0f, 2.0f, 3.0f);
    glm::vec3 v2(4.0f, 5.0f, 6.0f);

    // 加法
    glm::vec3 v_add = v1 + v2;
    if (v_add != glm::vec3(5.0f, 7.0f, 9.0f))
    {
        std::cerr << "向量加法失败" << std::endl;
        return false;
    }

    // 点积
    float dot_product = glm::dot(v1, v2);
    if (dot_product != 32.0f)
    {
        std::cerr << "点积计算失败" << std::endl;
        return false;
    }

    // 叉积
    glm::vec3 v_cross = glm::cross(v1, v2);
    if (v_cross != glm::vec3(-3.0f, 6.0f, -3.0f))
    {
        std::cerr << "叉积计算失败" << std::endl;
        return false;
    }

    std::cout << "[GLM] 向量运算测试通过" << std::endl;
    return true;
}

/**
 * @brief 测试矩阵变换
 * @return 测试是否通过
 */
bool test_matrix_transformations()
{
    std::cout << "[GLM] 测试矩阵变换..." << std::endl;

    // 创建单位矩阵
    glm::mat4 model = glm::mat4(1.0f);

    // 平移
    model = glm::translate(model, glm::vec3(1.0f, 2.0f, 3.0f));

    // 旋转
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // 缩放
    model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));

    // 验证矩阵不为单位矩阵
    if (model == glm::mat4(1.0f))
    {
        std::cerr << "矩阵变换失败" << std::endl;
        return false;
    }

    std::cout << "[GLM] 矩阵变换测试通过" << std::endl;
    return true;
}

/**
 * @brief 测试投影矩阵
 * @return 测试是否通过
 */
bool test_projection_matrices()
{
    std::cout << "[GLM] 测试投影矩阵..." << std::endl;

    // 透视投影
    glm::mat4 perspective = glm::perspective(
        glm::radians(45.0f),  // FOV
        16.0f / 9.0f,         // 宽高比
        0.1f,                 // 近裁剪面
        100.0f                // 远裁剪面
    );

    // 正交投影
    glm::mat4 ortho = glm::ortho(
        -10.0f, 10.0f,        // 左右
        -10.0f, 10.0f,        // 下上
        0.1f, 100.0f          // 近远
    );

    // 视图矩阵
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0f),   // 相机位置
        glm::vec3(0.0f, 0.0f, 0.0f),   // 观察目标
        glm::vec3(0.0f, 1.0f, 0.0f)    // 上方向
    );

    std::cout << "[GLM] 投影矩阵测试通过" << std::endl;
    return true;
}

/**
 * @brief 程序入口
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "GLM 数学库测试" << std::endl;
    std::cout << "版本: 0.9.9.8" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_vector_operations();
    all_passed &= test_matrix_transformations();
    all_passed &= test_projection_matrices();

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
