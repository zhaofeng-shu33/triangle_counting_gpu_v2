build/nvtc-variant: nvtc/main.cpp 
	nvcc -std=c++11 -DOPENMP=1 -DTRCOUNTING=1 -O3 -DNDEBUG -Xcompiler -fopenmp -Xptxas -O3 nvtc/main.cpp nvtc/mygraph.cpp nvtc/gpu.cu nvtc/gpu-thrust.cu nvtc/counting_cpu.cpp nvtc/graph.cpp -o nvtc-variant
