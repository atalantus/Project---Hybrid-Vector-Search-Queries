cmake_minimum_required(VERSION 3.20)

project(baseline_project VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-I. -O3 -mavx2)

add_executable(baseline.out src/test.cpp)
target_include_directories(baseline.out PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(optimized.out src/test.cpp)
target_include_directories(optimized.out PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(optimized.out PRIVATE OPT_IMPL)