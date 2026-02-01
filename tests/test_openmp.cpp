/**
 * @file test_openmp.cpp
 * @brief OpenMP并行计算测试
 * @details 测试并行循环、并行归约、线程管理等功能
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief 测试OpenMP并行循环
 * @return 测试是否通过
 */
bool test_parallel_loop()
{
    std::cout << "[OpenMP] 测试并行循环..." << std::endl;

#ifdef _OPENMP
    try
    {
        const int N = 1000000;
        std::vector<double> data(N);

        // 串行版本计时
        auto start_serial = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i)
        {
            data[i] = std::sin(i * 0.001) * std::cos(i * 0.002);
        }
        auto end_serial = std::chrono::high_resolution_clock::now();
        auto serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial).count();

        // 并行版本计时
        auto start_parallel = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            data[i] = std::sin(i * 0.001) * std::cos(i * 0.002);
        }
        auto end_parallel = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();

        std::cout << "串行时间: " << serial_time << " ms" << std::endl;
        std::cout << "并行时间: " << parallel_time << " ms" << std::endl;
        if (serial_time > 0)
        {
            std::cout << "加速比: " << (float)serial_time / parallel_time << "x" << std::endl;
        }

        std::cout << "[OpenMP] 并行循环测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "并行循环测试失败: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief 测试OpenMP并行归约
 * @return 测试是否通过
 */
bool test_parallel_reduction()
{
    std::cout << "[OpenMP] 测试并行归约..." << std::endl;

#ifdef _OPENMP
    try
    {
        const int N = 1000000;
        double sum = 0.0;

        // 并行归约求和
#pragma omp parallel for reduction(+:sum)
        for (int i = 1; i <= N; ++i)
        {
            sum += 1.0 / i;  // 调和级数
        }

        std::cout << "并行归约结果: " << sum << std::endl;

        // 验证结果（与串行版本比较）
        double serial_sum = 0.0;
        for (int i = 1; i <= N; ++i)
        {
            serial_sum += 1.0 / i;
        }

        if (std::abs(sum - serial_sum) > 1e-6)
        {
            std::cerr << "并行归约结果与串行结果不一致" << std::endl;
            return false;
        }

        std::cout << "[OpenMP] 并行归约测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "并行归约测试失败: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief 测试OpenMP线程信息
 * @return 测试是否通过
 */
bool test_thread_info()
{
    std::cout << "[OpenMP] 测试线程信息..." << std::endl;

#ifdef _OPENMP
    try
    {
        int max_threads = omp_get_max_threads();
        std::cout << "最大线程数: " << max_threads << std::endl;

#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();

#pragma omp critical
            {
                std::cout << "线程 " << thread_id << " / " << num_threads << " 正在运行" << std::endl;
            }
        }

        std::cout << "[OpenMP] 线程信息测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "线程信息测试失败: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief 测试OpenMP动态调度
 * @return 测试是否通过
 */
bool test_dynamic_schedule()
{
    std::cout << "[OpenMP] 测试动态调度..." << std::endl;

#ifdef _OPENMP
    try
    {
        const int N = 100;
        std::vector<int> results(N);

        // 使用动态调度
#pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < N; ++i)
        {
            // 模拟不同计算量的任务
            double sum = 0.0;
            for (int j = 0; j < i * 1000; ++j)
            {
                sum += std::sin(j * 0.001);
            }
            results[i] = i;
        }

        // 验证结果
        bool correct = true;
        for (int i = 0; i < N; ++i)
        {
            if (results[i] != i)
            {
                correct = false;
                break;
            }
        }

        if (!correct)
        {
            std::cerr << "动态调度结果错误" << std::endl;
            return false;
        }

        std::cout << "[OpenMP] 动态调度测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "动态调度测试失败: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief 程序入口
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "OpenMP 并行计算测试" << std::endl;
#ifdef _OPENMP
    std::cout << "OpenMP版本: " << _OPENMP << std::endl;
    std::cout << "最大线程数: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP未启用" << std::endl;
#endif
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_parallel_loop();
    all_passed &= test_parallel_reduction();
    all_passed &= test_thread_info();
    all_passed &= test_dynamic_schedule();

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
