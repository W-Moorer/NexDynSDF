/**
 * @file test_eigen.cpp
 * @brief Eigen3线性代数库测试
 * @details 测试矩阵运算、线性方程组求解、特征值计算等功能
 */

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

/**
 * @brief 测试矩阵基本运算
 * @return 测试是否通过
 */
bool test_matrix_operations()
{
    std::cout << "[Eigen] 测试矩阵基本运算..." << std::endl;

    try
    {
        // 创建矩阵
        Eigen::Matrix3f A;
        A << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f,
             7.0f, 8.0f, 10.0f;

        Eigen::Matrix3f B;
        B << 2.0f, 0.0f, 1.0f,
             1.0f, 2.0f, 0.0f,
             0.0f, 1.0f, 2.0f;

        // 矩阵加法
        Eigen::Matrix3f C = A + B;
        std::cout << "A + B =\n" << C << std::endl;

        // 矩阵乘法
        Eigen::Matrix3f D = A * B;
        std::cout << "A * B =\n" << D << std::endl;

        // 矩阵转置
        Eigen::Matrix3f At = A.transpose();
        std::cout << "A^T =\n" << At << std::endl;

        // 行列式
        float det = A.determinant();
        std::cout << "det(A) = " << det << std::endl;

        if (std::abs(det) < 1e-6f)
        {
            std::cerr << "矩阵A是奇异的" << std::endl;
            return false;
        }

        std::cout << "[Eigen] 矩阵基本运算测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "矩阵运算测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试线性方程组求解
 * @return 测试是否通过
 */
bool test_linear_solver()
{
    std::cout << "[Eigen] 测试线性方程组求解..." << std::endl;

    try
    {
        // 创建系数矩阵
        Eigen::Matrix3f A;
        A << 2.0f, -1.0f, 0.0f,
            -1.0f, 2.0f, -1.0f,
             0.0f, -1.0f, 2.0f;

        // 创建右侧向量
        Eigen::Vector3f b(1.0f, 0.0f, 1.0f);

        // 使用LU分解求解
        Eigen::Vector3f x = A.lu().solve(b);

        std::cout << "方程组 Ax = b" << std::endl;
        std::cout << "A =\n" << A << std::endl;
        std::cout << "b =\n" << b.transpose() << std::endl;
        std::cout << "x =\n" << x.transpose() << std::endl;

        // 验证解
        Eigen::Vector3f b_check = A * x;
        if (!b.isApprox(b_check, 1e-5f))
        {
            std::cerr << "解验证失败" << std::endl;
            return false;
        }

        std::cout << "[Eigen] 线性方程组求解测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "线性方程组求解测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试特征值和特征向量
 * @return 测试是否通过
 */
bool test_eigenvalues()
{
    std::cout << "[Eigen] 测试特征值计算..." << std::endl;

    try
    {
        // 对称矩阵
        Eigen::Matrix2f A;
        A << 4.0f, 1.0f,
             1.0f, 3.0f;

        // 计算特征值
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(A);

        if (solver.info() != Eigen::Success)
        {
            std::cerr << "特征值计算失败" << std::endl;
            return false;
        }

        std::cout << "矩阵 A =\n" << A << std::endl;
        std::cout << "特征值 =\n" << solver.eigenvalues() << std::endl;
        std::cout << "特征向量 =\n" << solver.eigenvectors() << std::endl;

        std::cout << "[Eigen] 特征值计算测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "特征值计算测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试动态矩阵
 * @return 测试是否通过
 */
bool test_dynamic_matrices()
{
    std::cout << "[Eigen] 测试动态矩阵..." << std::endl;

    try
    {
        // 创建动态矩阵
        Eigen::MatrixXd M = Eigen::MatrixXd::Random(5, 5);
        Eigen::VectorXd v = Eigen::VectorXd::Random(5);

        std::cout << "随机矩阵 M (5x5):\n" << M << std::endl;
        std::cout << "随机向量 v (5):\n" << v.transpose() << std::endl;

        // 矩阵向量乘法
        Eigen::VectorXd result = M * v;
        std::cout << "M * v =\n" << result.transpose() << std::endl;

        std::cout << "[Eigen] 动态矩阵测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "动态矩阵测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 程序入口
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "Eigen3 线性代数库测试" << std::endl;
    std::cout << "版本: 3.4.0" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_matrix_operations();
    all_passed &= test_linear_solver();
    all_passed &= test_eigenvalues();
    all_passed &= test_dynamic_matrices();

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
