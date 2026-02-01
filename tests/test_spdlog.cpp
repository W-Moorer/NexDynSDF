/**
 * @file test_spdlog.cpp
 * @brief spdlog logging library test
 * @details Test log output, formatting, level control, etc.
 */

#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/fmt/ostr.h>

/**
 * @brief Test basic logging output
 * @return Whether the test passed
 */
bool test_basic_logging()
{
    std::cout << "[spdlog] Testing basic logging..." << std::endl;

    try
    {
        // Different log levels
        spdlog::info("This is an info message");
        spdlog::warn("This is a warning message");
        spdlog::error("This is an error message");
        spdlog::debug("This is a debug message");

        std::cout << "[spdlog] Basic logging test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Basic logging test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test formatted logging
 * @return Whether the test passed
 */
bool test_formatted_logging()
{
    std::cout << "[spdlog] Testing formatted logging..." << std::endl;

    try
    {
        int value = 42;
        float pi = 3.14159f;
        std::string name = "NexDynSDF";

        spdlog::info("Project name: {}", name);
        spdlog::info("Integer value: {}, Float value: {:.2f}", value, pi);
        spdlog::info("Multiple params: {} {} {}", 1, 2, 3);

        std::cout << "[spdlog] Formatted logging test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Formatted logging test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test colored console output
 * @return Whether the test passed
 */
bool test_colored_console()
{
    std::cout << "[spdlog] Testing colored console output..." << std::endl;

    try
    {
        // Create colored console logger
        auto console = spdlog::stdout_color_mt("console");
        console->set_level(spdlog::level::info);
        console->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

        console->trace("Trace message");
        console->debug("Debug message");
        console->info("Info message");
        console->warn("Warning message");
        console->error("Error message");
        console->critical("Critical message");

        std::cout << "[spdlog] Colored console test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Colored console test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test log level control
 * @return Whether the test passed
 */
bool test_log_level()
{
    std::cout << "[spdlog] Testing log level control..." << std::endl;

    try
    {
        auto logger = spdlog::stdout_color_mt("level_test");

        // Set log level to warning
        logger->set_level(spdlog::level::warn);

        logger->trace("This will not show");
        logger->debug("This will not show");
        logger->info("This will not show");
        logger->warn("This will show (Warning)");
        logger->error("This will show (Error)");

        // Restore to info level
        logger->set_level(spdlog::level::info);
        logger->info("Level restored to info");

        std::cout << "[spdlog] Log level control test passed" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Log level test failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Program entry point
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "spdlog Library Test" << std::endl;
    std::cout << "Version: 1.12.0" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_basic_logging();
    all_passed &= test_formatted_logging();
    all_passed &= test_colored_console();
    all_passed &= test_log_level();

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
