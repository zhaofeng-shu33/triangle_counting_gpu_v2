build/nvtc-variant: nvtc/main.cpp 
	nvcc -std=c++11 -DOPENMP=1 -DTRCOUNTING=1 -O3 -DNDEBUG -Xcompiler -fopenmp -Xptxas -O3 nvtc/main.cpp nvtc/TrCountingGraph.cpp nvtc/gpu.cu nvtc/gpu-thrust.cu -o nvtc-variant
