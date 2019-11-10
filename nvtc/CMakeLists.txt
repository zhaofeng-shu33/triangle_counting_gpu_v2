add_library(nvtc_variant gpu.cu graph.cpp timer.cpp cpu.cpp gpu-thrust.cu)
add_executable(nvtc-variant main.cpp)
target_link_libraries(nvtc-variant nvtc_variant)
