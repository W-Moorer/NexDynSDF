/**
 * @file test_openmp.cpp
 * @brief OpenMP parallel computing test
 * @details Tests parallel loops, parallel reduction, thread management, etc.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Test OpenMP parallel loop
 * @return Whether the test passed
 */
bool test_parallel_loop()
{
    std::cout << "[OpenMP] Testing parallel loop..." << std::endl;

#ifdef _OPENMP
    try
    {
        const int N = 1000000;
        std::vector<double> data(N);

        // Serial version timing
        auto start_serial = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i)
        {
            data[i] = std::sin(i * 0.001) * std::cos(i * 0.002);
        }
        auto end_serial = std::chrono::high_resolution_clock::now();
        auto serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial).count();

        // Parallel version timing
        auto start_parallel = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            data[i] = std::sin(i * 0.001) * std::cos(i * 0.002);
        }
        auto end_parallel = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();

        std::cout << "Serial time: " << serial_time << " ms" << std::endl;
        std::cout << "Parallel time: " << parallel_time << " ms" << std::endl;
        if (serial_time > 0)
        {
            std::cout << "Speedup: " << (float)serial_time / parallel_time << "x" << std::endl;
        }

        std::cout << "[OpenMP] Parallel loop test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Parallel loop test failed: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief Test OpenMP parallel reduction
 * @return Whether the test passed
 */
bool test_parallel_reduction()
{
    std::cout << "[OpenMP] Testing parallel reduction..." << std::endl;

#ifdef _OPENMP
    try
    {
        const int N = 1000000;
        double sum = 0.0;

        // Parallel reduction sum
#pragma omp parallel for reduction(+:sum)
        for (int i = 1; i <= N; ++i)
        {
            sum += 1.0 / i;  // Harmonic series
        }

        std::cout << "Parallel reduction result: " << sum << std::endl;

        // Verify result (compare with serial version)
        double serial_sum = 0.0;
        for (int i = 1; i <= N; ++i)
        {
            serial_sum += 1.0 / i;
        }

        if (std::abs(sum - serial_sum) > 1e-6)
        {
            std::cerr << "Parallel reduction result inconsistent with serial result" << std::endl;
            return false;
        }

        std::cout << "[OpenMP] Parallel reduction test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Parallel reduction test failed: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief Test OpenMP thread info
 * @return Whether the test passed
 */
bool test_thread_info()
{
    std::cout << "[OpenMP] Testing thread info..." << std::endl;

#ifdef _OPENMP
    try
    {
        int max_threads = omp_get_max_threads();
        std::cout << "Max threads: " << max_threads << std::endl;

#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();

#pragma omp critical
            {
                std::cout << "Thread " << thread_id << " / " << num_threads << " is running" << std::endl;
            }
        }

        std::cout << "[OpenMP] Thread info test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Thread info test failed: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief Test OpenMP dynamic schedule
 * @return Whether the test passed
 */
bool test_dynamic_schedule()
{
    std::cout << "[OpenMP] Testing dynamic schedule..." << std::endl;

#ifdef _OPENMP
    try
    {
        const int N = 100;
        std::vector<int> results(N);

        // Use dynamic schedule
#pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < N; ++i)
        {
            // Simulate tasks with different computation costs
            double sum = 0.0;
            for (int j = 0; j < i * 1000; ++j)
            {
                sum += std::sin(j * 0.001);
            }
            results[i] = i;
        }

        // Verify results
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
            std::cerr << "Dynamic schedule result error" << std::endl;
            return false;
        }

        std::cout << "[OpenMP] Dynamic schedule test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Dynamic schedule test failed: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[OpenMP] 未启用OpenMP支持" << std::endl;
    return true;
#endif
}

/**
 * @brief Program entry
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "OpenMP Parallel Computing Test" << std::endl;
#ifdef _OPENMP
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP not enabled" << std::endl;
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
        std::cout << "All tests passed!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}
