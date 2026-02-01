/**
 * @file main.cpp
 * @brief NexDynSDF 项目主程序
 * @details 测试所有依赖库的加载和功能
 */

#include <iostream>
#include <vector>

// GLM - 数学库（向量、矩阵运算）
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// spdlog - 日志输出
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// cereal - 序列化
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>

// Eigen3 - 线性代数
#include <Eigen/Dense>
#include <Eigen/LU>

// OpenMP - 并行计算
#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief 测试GLM数学库
 * @details 测试向量运算和矩阵变换
 */
void test_glm()
{
    spdlog::info("===== 测试 GLM 数学库 =====");

    // 向量运算
    glm::vec3 v1(1.0f, 2.0f, 3.0f);
    glm::vec3 v2(4.0f, 5.0f, 6.0f);
    glm::vec3 v3 = v1 + v2;

    spdlog::info("向量 v1: ({}, {}, {})", v1.x, v1.y, v1.z);
    spdlog::info("向量 v2: ({}, {}, {})", v2.x, v2.y, v2.z);
    spdlog::info("v1 + v2 = ({}, {}, {})", v3.x, v3.y, v3.z);

    // 矩阵变换
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.0f, 2.0f, 3.0f));
    model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));

    spdlog::info("模型矩阵创建成功");
}

/**
 * @brief 测试cereal序列化库
 * @details 测试数据的序列化和反序列化
 */
void test_cereal()
{
    spdlog::info("===== 测试 Cereal 序列化库 =====");

    // 准备测试数据
    std::vector<float> original_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // 序列化到内存
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive archive(ss);
        archive(original_data);
    }

    spdlog::info("序列化数据大小: {} 字节", ss.str().size());

    // 从内存反序列化
    std::vector<float> loaded_data;
    {
        cereal::BinaryInputArchive archive(ss);
        archive(loaded_data);
    }

    spdlog::info("反序列化数据: ");
    for (size_t i = 0; i < loaded_data.size(); ++i)
    {
        spdlog::info("  [{}] = {}", i, loaded_data[i]);
    }
}

/**
 * @brief 测试Eigen3线性代数库
 * @details 测试矩阵运算和求解
 */
void test_eigen()
{
    spdlog::info("===== 测试 Eigen3 线性代数库 =====");

    // 创建矩阵
    Eigen::Matrix3f A;
    A << 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 10.0f;

    spdlog::info("矩阵 A:");
    for (int i = 0; i < 3; ++i)
    {
        spdlog::info("  {:.2f} {:.2f} {:.2f}", A(i, 0), A(i, 1), A(i, 2));
    }

    // 计算行列式
    float det = A.determinant();
    spdlog::info("矩阵 A 的行列式: {}", det);

    // 求解线性方程组 Ax = b
    Eigen::Vector3f b(1.0f, 2.0f, 3.0f);
    Eigen::Vector3f x = A.lu().solve(b);

    spdlog::info("求解 Ax = b:");
    spdlog::info("  b = ({}, {}, {})", b(0), b(1), b(2));
    spdlog::info("  x = ({}, {}, {})", x(0), x(1), x(2));
}

/**
 * @brief 测试OpenMP并行计算
 * @details 测试并行循环加速
 */
void test_openmp()
{
    spdlog::info("===== 测试 OpenMP 并行计算 =====");

#ifdef WITH_OPENMP
    int num_threads = omp_get_max_threads();
    spdlog::info("OpenMP 最大线程数: {}", num_threads);

    // 并行计算示例
    const int N = 1000000;
    std::vector<double> data(N);

    double start_time = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        data[i] = std::sin(i * 0.001) * std::cos(i * 0.002);
    }

    double end_time = omp_get_wtime();
    spdlog::info("并行计算完成，耗时: {:.4f} 秒", end_time - start_time);
#else
    spdlog::warn("OpenMP 未启用");
#endif
}

/**
 * @brief 程序入口点
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 程序退出码
 */
int main(int argc, char **argv)
{
    // 配置spdlog日志
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    auto logger = std::make_shared<spdlog::logger>("nexdynsdf", console_sink);
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);

    spdlog::info("========================================");
    spdlog::info("NexDynSDF 项目启动");
    spdlog::info("版本: 0.1.0");
    spdlog::info("========================================");

    // 测试各个库
    try
    {
        test_glm();
        test_cereal();
        test_eigen();
        test_openmp();
    }
    catch (const std::exception &e)
    {
        spdlog::error("测试过程中发生错误: {}", e.what());
        return 1;
    }

    spdlog::info("========================================");
    spdlog::info("所有测试完成！");
    spdlog::info("========================================");

    return 0;
}
