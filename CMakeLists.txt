cmake_minimum_required(VERSION 3.20)

project(hybrid_vector_search_queries VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-I. -O3 -mavx2)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
else ()
    add_compile_options(-O0 -g -Wall -Wpadded -Wpacked -Wpacked-not-aligned)
endif ()

add_executable(baseline.out src/test.cpp)
target_include_directories(baseline.out PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(baseline.out PRIVATE IMPL=1)

add_executable(optimized.out src/test.cpp)
target_include_directories(optimized.out PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(optimized.out PRIVATE IMPL=2)

add_executable(changes.out src/test.cpp)
target_include_directories(changes.out PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(changes.out PRIVATE IMPL=3)