#pragma once
#include <stdint.h>
#include <mutex>

using namespace std;


class TrCountingGraph{
	public:
		// Construct Function
		TrCountingGraph(const char* file_name);

		// node ID -> neighboor table offset from int* neighboor.
		int64_t* offset;

		char* entire_data;
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
};

int64_t get_split_v2(int64_t* offset, int nodeid_max, int split_num, int64_t*& out);
void cpu_counting_edge_first_v2(TrCountingGraph* g, int64_t offset_start, int64_t* out);
void sort_neighboor(TrCountingGraph* g, int* d);