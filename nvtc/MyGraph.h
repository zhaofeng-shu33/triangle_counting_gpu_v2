#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <ctime>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <mutex>
using namespace std;


class MyGraph{
	public:
		// Construct Function
		MyGraph(const char* file_name);

		// node ID -> neighboor table offset from int* neighboor.
		int64_t* offset;

		// node ID -> Node degree.
		int* degree;

		// neighboor table starting address
		int* neighboor;
		int* neighboor_start;

		// maximum node id
		int64_t nodeid_max;

		// total number of edges
		int64_t edge_num;

		//mutex* lock;
		mutex fin_lock;

	private:
		void sort_neighboor(int* d);
};

int64_t get_split_v2(int64_t* offset, int nodeid_max, int split_num, int64_t cpu_offset, int64_t*& out);
void cpu_counting_edge_first_v2(MyGraph* g, int64_t cpu_offset, int64_t* out);