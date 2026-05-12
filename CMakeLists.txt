cmake_minimum_required(VERSION 3.20)
project(krylov_option_pricing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Find Eigen
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${CMAKE_SOURCE_DIR}/eigen)

# Main library
add_library(option_pricing_lib INTERFACE)
target_include_directories(option_pricing_lib INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(option_pricing_lib INTERFACE eigen)

# Main executable
add_executable(option_pricing src/main.cpp)
target_link_libraries(option_pricing PRIVATE option_pricing_lib)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(option_pricing PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(option_pricing PRIVATE USE_OPENMP)
endif()
