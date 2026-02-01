/**
 * @file Timer.h
 * @brief High-resolution timer for performance measurement
 */

#pragma once

#include <chrono>

namespace sdflib
{

class Timer
{
public:
    Timer();
    
    void start();
    double getElapsedSeconds() const;
    double getElapsedMilliseconds() const;

private:
    std::chrono::high_resolution_clock::time_point mStartTime;
};

}
