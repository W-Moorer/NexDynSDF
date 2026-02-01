/**
 * @file Timer.cpp
 * @brief High-resolution timer implementation
 */

#include "sdflib/utils/Timer.h"

namespace sdflib
{

Timer::Timer()
{
    start();
}

void Timer::start()
{
    mStartTime = std::chrono::high_resolution_clock::now();
}

double Timer::getElapsedSeconds() const
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - mStartTime);
    return duration.count() / 1000000.0;
}

double Timer::getElapsedMilliseconds() const
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - mStartTime);
    return duration.count() / 1000.0;
}

}
