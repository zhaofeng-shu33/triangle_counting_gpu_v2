#pragma once
#include <nvtc/config.h>
#include <stdio.h>
#include <stdint.h>
#include <nvtc/config.h>
using namespace std;


class TrCountingGraph {
	public:
		// Construct Function
		TrCountingGraph(const char* file_name);
        ~TrCountingGraph();
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

};

void get_i_j(int n, int ij, int* i, int* j);
int64_t get_split_v2(int64_t* offset, int nodeid_max, int split_num, int64_t*& out);
void cpu_counting_edge_first_v2(TrCountingGraph* g, int64_t offset_start, int64_t offset_end, int64_t* out);
void sort_neighboor(TrCountingGraph* g);
int64_t get_edge_num(FILE* file);
int get_max_id(int* data, int64_t len);
