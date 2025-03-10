cmake_minimum_required(VERSION 3.10)
project(MatrixMultiply)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# + Find and enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multiply PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(matrix_multiply PRIVATE ${OpenMP_CXX_FLAGS})
endif()

add_executable(matrix_multiply main.cpp)