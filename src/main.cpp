/**
 * @file main.cpp
 * @brief NexDynSDF project main program
 * @details Tests the loading and functionality of all dependency libraries
 */

#include <iostream>
#include <vector>

// GLM - Math library (vectors, matrices)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// spdlog - Logging
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// cereal - Serialization
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>

// Eigen3 - Linear algebra
#include <Eigen/Dense>
#include <Eigen/LU>

// OpenMP - Parallel computing
#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Test GLM math library
 * @details Tests vector operations and matrix transformations
 */
void test_glm()
{
    spdlog::info("===== Test GLM Math Library =====");

    // Vector operations
    glm::vec3 v1(1.0f, 2.0f, 3.0f);
    glm::vec3 v2(4.0f, 5.0f, 6.0f);
    glm::vec3 v3 = v1 + v2;

    spdlog::info("Vector v1: ({}, {}, {})", v1.x, v1.y, v1.z);
    spdlog::info("Vector v2: ({}, {}, {})", v2.x, v2.y, v2.z);
    spdlog::info("v1 + v2 = ({}, {}, {})", v3.x, v3.y, v3.z);

    // Matrix transformations
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.0f, 2.0f, 3.0f));
    model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));

    spdlog::info("Model matrix created successfully");
}

/**
 * @brief Test cereal serialization library
 * @details Tests data serialization and deserialization
 */
void test_cereal()
{
    spdlog::info("===== Test Cereal Serialization Library =====");

    // Prepare test data
    std::vector<float> original_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Serialize to memory
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive archive(ss);
        archive(original_data);
    }

    spdlog::info("Serialized data size: {} bytes", ss.str().size());

    // Deserialize from memory
    std::vector<float> loaded_data;
    {
        cereal::BinaryInputArchive archive(ss);
        archive(loaded_data);
    }

    spdlog::info("Deserialized data: ");
    for (size_t i = 0; i < loaded_data.size(); ++i)
    {
        spdlog::info("  [{}] = {}", i, loaded_data[i]);
    }
}

/**
 * @brief Test Eigen3 linear algebra library
 * @details Tests matrix operations and solving
 */
void test_eigen()
{
    spdlog::info("===== Test Eigen3 Linear Algebra Library =====");

    // Create matrix
    Eigen::Matrix3f A;
    A << 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 10.0f;

    spdlog::info("Matrix A:");
    for (int i = 0; i < 3; ++i)
    {
        spdlog::info("  {:.2f} {:.2f} {:.2f}", A(i, 0), A(i, 1), A(i, 2));
    }

    // Calculate determinant
    float det = A.determinant();
    spdlog::info("Determinant of matrix A: {}", det);

    // Solve linear system Ax = b
    Eigen::Vector3f b(1.0f, 2.0f, 3.0f);
    Eigen::Vector3f x = A.lu().solve(b);

    spdlog::info("Solving Ax = b:");
    spdlog::info("  b = ({}, {}, {})", b(0), b(1), b(2));
    spdlog::info("  x = ({}, {}, {})", x(0), x(1), x(2));
}

/**
 * @brief Test OpenMP parallel computing
 * @details Tests parallel loop acceleration
 */
void test_openmp()
{
    spdlog::info("===== Test OpenMP Parallel Computing =====");

#ifdef WITH_OPENMP
    int num_threads = omp_get_max_threads();
    spdlog::info("OpenMP max threads: {}", num_threads);

    // Parallel computing example
    const int N = 1000000;
    std::vector<double> data(N);

    double start_time = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        data[i] = std::sin(i * 0.001) * std::cos(i * 0.002);
    }

    double end_time = omp_get_wtime();
    spdlog::info("Parallel computation finished, elapsed time: {:.4f} seconds", end_time - start_time);
#else
    spdlog::warn("OpenMP not enabled");
#endif
}

/**
 * @brief Program entry point
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments array
 * @return Program exit code
 */
int main(int argc, char **argv)
{
    // Configure spdlog logging
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    auto logger = std::make_shared<spdlog::logger>("nexdynsdf", console_sink);
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);

    spdlog::info("========================================");
    spdlog::info("NexDynSDF Project Started");
    spdlog::info("Version: 0.1.0");
    spdlog::info("========================================");

    // Test each library
    try
    {
        test_glm();
        test_cereal();
        test_eigen();
        test_openmp();
    }
    catch (const std::exception &e)
    {
        spdlog::error("Error occurred during testing: {}", e.what());
        return 1;
    }

    spdlog::info("========================================");
    spdlog::info("All tests completed!");
    spdlog::info("========================================");

    return 0;
}
