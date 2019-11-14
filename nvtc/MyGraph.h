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
#define BUFFERSIZE 8192
#define BATCHSIZE BUFFERSIZE/8

using namespace std;

class MyGraph{
	public:
		// Construct Function
		MyGraph(const char* file_name);
		
		//Query arc exist using go-through check.
		bool arc_exist(int u, int v);

		//Query arc exist using binary search.
		bool arc_exist_sorted(int u, int v);

		// node ID -> neighboor table offset from int* neighboor.
		int64_t* offset;

		// node ID -> Node degree.
		int* degree;

		// node ID -> array length starting from offset.
		int* length;

		// neighboor table starting address
		int* neighboor;

		// maximum node id
		int64_t nodeid_max;

		// total number of edges
		int64_t edge_num;

	private:
		void sort_neighboor();
		bool inner_arc_exist(int u, int v, int* d);
};