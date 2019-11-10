cmake_minimum_required(VERSION 3.13)
project(triangle_counting_gpu_variant CUDA CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(nvtc)