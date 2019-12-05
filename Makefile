build/nvtc-variant: nvtc/main.cpp 
	nvcc -I./build -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -DOPENMP=1 -DTRCOUNTING=1 -O3 -DNDEBUG -Xcompiler -fopenmp -Xptxas -O3 nvtc/main.cpp nvtc/TrCountingGraph.cpp nvtc/gpu.cu nvtc/gpu-thrust.cu -o nvtc-variant
