/**
 * @file test_all.cpp
 * @brief Summary test for all libraries
 * @details Verify all dependencies can be loaded and used
 */

#include <iostream>
#include <vector>
#include <string>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// spdlog
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// cereal
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>

// Eigen3
#include <Eigen/Dense>

// Enoki
#include <enoki/array.h>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Test result structure
 */
struct TestResult
{
    std::string name;
    bool passed;
    std::string message;
};

/**
 * @brief Test GLM library
 */
TestResult test_glm()
{
    TestResult result{"GLM", false, ""};
    try
    {
        glm::vec3 v1(1.0f, 2.0f, 3.0f);
        glm::vec3 v2(4.0f, 5.0f, 6.0f);
        glm::vec3 v3 = v1 + v2;

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(1.0f, 2.0f, 3.0f));

        result.passed = true;
        result.message = "Vector and matrix operations OK";
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Error: ") + e.what();
    }
    return result;
}

/**
 * @brief Test spdlog library
 */
TestResult test_spdlog()
{
    TestResult result{"spdlog", false, ""};
    try
    {
        auto console = spdlog::stdout_color_mt("test_console");
        console->set_level(spdlog::level::info);
        console->info("spdlog test message");

        result.passed = true;
        result.message = "Logging output OK";
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Error: ") + e.what();
    }
    return result;
}

/**
 * @brief Test cereal library
 */
TestResult test_cereal()
{
    TestResult result{"cereal", false, ""};
    try
    {
        std::vector<int> data = {1, 2, 3, 4, 5};
        std::stringstream ss;

        {
            cereal::BinaryOutputArchive archive(ss);
            archive(data);
        }

        std::vector<int> loaded;
        {
            cereal::BinaryInputArchive archive(ss);
            archive(loaded);
        }

        result.passed = (data == loaded);
        result.message = result.passed ? "Serialization OK" : "Data mismatch";
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Error: ") + e.what();
    }
    return result;
}

/**
 * @brief Test Eigen3 library
 */
TestResult test_eigen()
{
    TestResult result{"Eigen3", false, ""};
    try
    {
        Eigen::Matrix3f A;
        A << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f,
             7.0f, 8.0f, 10.0f;

        float det = A.determinant();
        Eigen::Vector3f b(1.0f, 2.0f, 3.0f);
        Eigen::Vector3f x = A.lu().solve(b);

        result.passed = true;
        result.message = "Matrix operations OK";
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Error: ") + e.what();
    }
    return result;
}

/**
 * @brief Test Enoki library
 */
TestResult test_enoki()
{
    TestResult result{"Enoki", false, ""};
    try
    {
        using namespace enoki;
        Array<float, 4> a(1.0f, 2.0f, 3.0f, 4.0f);
        Array<float, 4> b(5.0f, 6.0f, 7.0f, 8.0f);
        Array<float, 4> c = a + b;
        float sum = hsum(a);

        result.passed = true;
        result.message = "SIMD vectorization OK";
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Error: ") + e.what();
    }
    return result;
}

/**
 * @brief Test OpenMP
 */
TestResult test_openmp()
{
    TestResult result{"OpenMP", false, ""};
    try
    {
#ifdef _OPENMP
        int sum = 0;
        const int N = 1000;

#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; ++i)
        {
            sum += i;
        }

        int expected = N * (N - 1) / 2;
        result.passed = (sum == expected);
        result.message = result.passed ? "Parallel computing OK" : "Result error";
#else
        result.passed = true;
        result.message = "OpenMP not enabled";
#endif
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Error: ") + e.what();
    }
    return result;
}

/**
 * @brief Print test results table
 */
void print_results(const std::vector<TestResult>& results)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Library\t\tStatus\tDescription" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    int passed = 0;
    int failed = 0;

    for (const auto& result : results)
    {
        std::string status = result.passed ? "[PASS]" : "[FAIL]";
        std::cout << result.name << "\t\t" << status << "\t" << result.message << std::endl;

        if (result.passed)
            passed++;
        else
            failed++;
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Passed: " << passed << " / " << results.size() << std::endl;
    std::cout << "Failed: " << failed << " / " << results.size() << std::endl;
    std::cout << "========================================" << std::endl;
}

/**
 * @brief Program entry point
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "NexDynSDF Dependency Library Summary Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::vector<TestResult> results;

    // Run all tests
    std::cout << "Testing libraries..." << std::endl;

    results.push_back(test_glm());
    results.push_back(test_spdlog());
    results.push_back(test_cereal());
    results.push_back(test_eigen());
    results.push_back(test_enoki());
    results.push_back(test_openmp());

    // Print results
    print_results(results);

    // Check if all passed
    bool all_passed = true;
    for (const auto& result : results)
    {
        if (!result.passed)
        {
            all_passed = false;
            break;
        }
    }

    if (all_passed)
    {
        std::cout << "\nAll libraries loaded successfully! Project configured correctly." << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nSome libraries failed to load, please check configuration." << std::endl;
        return 1;
    }
}
