cmake_minimum_required(VERSION 3.20)
project(krylov_option_pricing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

# Default single-configuration generators to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(OPTION_PRICING_NATIVE_ARCH
       "Enable host-specific instruction generation for local timing runs"
       OFF)

# Find Eigen, with a pinned source fallback for clean checkouts.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# Main library
add_library(option_pricing_lib INTERFACE)
target_include_directories(option_pricing_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(option_pricing_lib INTERFACE Eigen3::Eigen)
target_compile_options(option_pricing_lib INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
if(OPTION_PRICING_NATIVE_ARCH)
    target_compile_options(option_pricing_lib INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>)
endif()

# Main executable
add_executable(option_pricing src/main.cpp)
target_link_libraries(option_pricing PRIVATE option_pricing_lib)

if(BUILD_TESTING)
    add_executable(solver_order_test tests/solver_order_test.cpp)
    target_link_libraries(solver_order_test PRIVATE option_pricing_lib)
    add_test(NAME solver_order COMMAND solver_order_test)

    add_executable(numerical_utilities_test tests/numerical_utilities_test.cpp)
    target_link_libraries(numerical_utilities_test PRIVATE option_pricing_lib)
    add_test(NAME numerical_utilities COMMAND numerical_utilities_test)
endif()
