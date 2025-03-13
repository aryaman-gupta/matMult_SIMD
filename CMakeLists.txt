cmake_minimum_required(VERSION 3.10)
project(MatrixMultiply)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

option(USE_O3 "Enable -O3 optimization" OFF)
option(USE_AVX2 "Enable AVX2 instructions" OFF)


if(USE_O3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

if(USE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

message("Using CXX flags: ${CMAKE_CXX_FLAGS}")

add_executable(matrix_multiply main.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multiply PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(matrix_multiply PRIVATE ${OpenMP_CXX_FLAGS})
endif()