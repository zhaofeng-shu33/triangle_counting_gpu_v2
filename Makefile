build/nvtc-cpu: nvtc/main.cpp
	g++ -DOPENMP=1 -DTIMECOUNTING=1 -DTRCOUNTING=1 -DVERBOSE=1 -D_GLIBCXX_PARALLEL -O2 -DNDEBUG -fopenmp -std=c++11 nvtc/main.cpp nvtc/graph.cpp nvtc/counting_cpu.cpp nvtc/timer.cpp -o build/nvtc-cpu
