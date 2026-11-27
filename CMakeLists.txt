cmake_minimum_required(VERSION 3.20)
project(krylov_option_pricing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Find Eigen, with a pinned source fallback for clean checkouts.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# Main library
add_library(option_pricing_lib INTERFACE)
target_include_directories(option_pricing_lib INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(option_pricing_lib INTERFACE Eigen3::Eigen)

# Main executable
add_executable(option_pricing src/main.cpp)
target_link_libraries(option_pricing PRIVATE option_pricing_lib)

if(BUILD_TESTING)
    add_executable(solver_order_test tests/solver_order_test.cpp)
    target_link_libraries(solver_order_test PRIVATE option_pricing_lib)
    add_test(NAME solver_order COMMAND solver_order_test)
endif()
