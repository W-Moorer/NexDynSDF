cmake_minimum_required(VERSION 3.20)

# Set FetchContent cache directory BEFORE including FetchContent module
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH "FetchContent cache directory")

# Project info
project(NexDynSDF 
    VERSION 0.1.0 
    DESCRIPTION "Dynamic SDF computation project" 
    LANGUAGES CXX
)

# C++ standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent module
include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

# Enable testing
enable_testing()

# ============ Dependency Configuration ============

# Use vcpkg manifest mode - dependencies are declared in vcpkg.json
# vcpkg will automatically install them when cmake configures
find_package(glm CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# Enoki - Vectorization optimization (vcpkg not available, use FetchContent with cache)
FetchContent_Declare(enoki_lib
    GIT_REPOSITORY https://github.com/mitsuba-renderer/enoki.git
    GIT_TAG 2a18afa
)
FetchContent_GetProperties(enoki_lib)
if(NOT enoki_lib_POPULATED)
    FetchContent_Populate(enoki_lib)
    add_library(enoki INTERFACE)
    target_include_directories(enoki INTERFACE $<BUILD_INTERFACE:${enoki_lib_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
endif()

# FCPW - Fast closest point query (vcpkg not available, use FetchContent with cache)
FetchContent_Declare(fcpw_lib
    GIT_REPOSITORY https://github.com/rohan-sawhney/fcpw.git
    GIT_TAG dd65ec2
)
FetchContent_GetProperties(fcpw_lib)
if(NOT fcpw_lib_POPULATED)
    FetchContent_Populate(fcpw_lib)
    add_library(fcpw INTERFACE)
    target_include_directories(fcpw INTERFACE $<BUILD_INTERFACE:${fcpw_lib_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# ============ Library Target ============

add_library(sdflib STATIC
    src/SdfFunction.cpp
    src/OctreeSdf.cpp
    src/OctreeSdfUniform.cpp
    src/ExactOctreeSdf.cpp
    src/utils/Mesh.cpp
    src/utils/Timer.cpp
    src/utils/GJK.cpp
    src/utils/TriangleUtils.cpp
)

target_include_directories(sdflib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${enoki_lib_SOURCE_DIR}/include
    ${fcpw_lib_SOURCE_DIR}
)

target_link_libraries(sdflib PUBLIC
    glm::glm
    spdlog::spdlog
    cereal::cereal
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

target_compile_features(sdflib PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(sdflib PUBLIC /arch:AVX2)
else()
    target_compile_options(sdflib PUBLIC -march=native)
endif()

# ============ SdfExporter Tool ============

add_executable(SdfExporter
    tools/SdfExporter/main.cpp
)

target_link_libraries(SdfExporter PRIVATE sdflib)

# ============ SdfSampler Tool ============

add_executable(SdfSampler
    tools/SdfSampler/main.cpp
)

target_link_libraries(SdfSampler PRIVATE sdflib)

# ============ Test Executable ============

add_executable(test_all tests/test_all.cpp)
target_link_libraries(test_all PRIVATE sdflib)
add_test(NAME all_test COMMAND test_all)

add_executable(test_nagata_enhanced tests/test_nagata_enhanced.cpp)
target_link_libraries(test_nagata_enhanced PRIVATE sdflib)
add_test(NAME nagata_enhanced_test COMMAND test_nagata_enhanced)

# Output configuration info
message(STATUS "========================================")
message(STATUS "NexDynSDF Project Configuration Complete")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "FetchContent Cache: ${FETCHCONTENT_BASE_DIR}")
message(STATUS "========================================")
