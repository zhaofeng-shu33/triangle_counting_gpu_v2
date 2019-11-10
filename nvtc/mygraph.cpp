#include "MyGraph.h"
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


MyGraph::MyGraph(const char* file_name){
	// Temporal variables
    std::ifstream fin;
	char buffer[BUFFERSIZE];
	char u_array[4], v_array[4];
	int64_t counter = 0;
	int *u, *v;
	int x, y;
	int node_max = 0;

	// Compute edge num by file length
	fin.open(file_name, ifstream::binary | ifstream::in);
	fin.seekg(0, fin.end);
	edge_num = fin.tellg()/8;
	fin.seekg(0, fin.beg);
	
	//Round 1, Get max id
	while (counter + BATCHSIZE < edge_num) {
		fin.read(buffer, BUFFERSIZE);
		u = reinterpret_cast<int*>(buffer);
		for (int j = 0; j < BATCHSIZE; j++) {
			x = *(u + 2 * j);
			y = *(u + 2 * j + 1);
			if (x > node_max) {
				node_max = x;
			}
			if (y > node_max) {
				node_max = y;
			}
		}
		counter = counter + BATCHSIZE;
	}
	for (int64_t i = counter; i < edge_num; i++) {
		fin.read(u_array, 4);
		fin.read(v_array, 4);
		u = reinterpret_cast<int*>(u_array);
		v = reinterpret_cast<int*>(v_array);
		x = *u;
		y = *v;
		if (x > node_max) {
			node_max = x;
		}
		if (y > node_max) {
			node_max = y;
		}
	}
	nodeid_max = node_max;
	//cout << "Max Node ID in dataset: " << node_max << endl;
	
	// Call for mem
	offset = new int64_t[nodeid_max +2];
	degree = new int[nodeid_max + 1];
	neighboor = new int[2 * edge_num + 1];
	int* _temp = new int[nodeid_max + 1];

	//Round 2, Get node degree
	fin.seekg(0, fin.beg);
	counter = 0;
	while (counter + BATCHSIZE < edge_num) {
		fin.read(buffer, BUFFERSIZE);
		u = reinterpret_cast<int*>(buffer);
		for (int j = 0; j < BATCHSIZE; j++) {
			x = *(u + 2 * j);
			y = *(u + 2 * j + 1);
			degree[x]++;
			degree[y]++;
		}
		counter = counter + BATCHSIZE;
	}
	for (int64_t i = counter; i < edge_num; i++) {
		fin.read(u_array, 4);
		fin.read(v_array, 4);
		u = reinterpret_cast<int*>(u_array);
		v = reinterpret_cast<int*>(v_array);
		x = *u;
		y = *v;
		degree[x]++;
		degree[y]++;
	}

	offset[0] = 0;
	for (int64_t i = 1; i <= nodeid_max+1; i++) {
		offset[i] = offset[i - 1] + degree[i - 1];
	}

	//Round 3, Record neighboors
	fin.seekg(0, fin.beg);
	counter = 0;
	while (counter + BATCHSIZE < edge_num) {
		fin.read(buffer, BUFFERSIZE);
		u = reinterpret_cast<int*>(buffer);
//#pragma omp parallel for 
		for (int j = 0; j < BATCHSIZE; j++) {
			x = *(u + 2 * j);
			y = *(u + 2 * j + 1);
			neighboor[offset[x] + _temp[x]++] = y;
			neighboor[offset[y] + _temp[y]++] = x;
			//neighboor[offset[*(u + 2 * j)] + _temp[*(u + 2 * j)]++] = *(u + 2 * j + 1);
			//neighboor[offset[*(u + 2 * j + 1)] + _temp[*(u + 2 * j + 1)]++] = *(u + 2 * j);
		}
		counter = counter + BATCHSIZE;
	}
	for (int64_t i = counter; i < edge_num; i++) {
		fin.read(u_array, 4);
		fin.read(v_array, 4);
		u = reinterpret_cast<int*>(u_array);
		v = reinterpret_cast<int*>(v_array);
		x = *u;
		y = *v;
		neighboor[offset[x] + _temp[x]++] = y;
		neighboor[offset[y] + _temp[y]++] = x;
	}
	//for (int64_t i = 0; i <= nodeid_max; i++) {
	//	if (degree[i] != _temp[i]) {
	//		cout << "error at " << i << endl;
	//		break;
	//	}
			
	//}

	sort_neighboor();
}

bool MyGraph::arc_exist(int u, int v) {
	int x, y;
	if (degree[u] < degree[v]) {
		x = u;
		y = v;
	}
	else {
		x = v;
		y = u;
	}
	for (int i = 0; i < degree[x]; i++) {
		if (neighboor[offset[x] + i] == y) {
			return true;
		}
	}
	return false;
}

void MyGraph::sort_neighboor() {
#pragma omp parallel for
	for (int64_t i = 0; i <= nodeid_max; i++) {
		sort(neighboor + offset[i], neighboor + offset[i + 1]);
	}
}

bool MyGraph::arc_exist_sorted(int u, int v) {
	int x, y;
	if (degree[u] < degree[v]) {
		x = u;
		y = v;
	}
	else {
		x = v;
		y = u;
	}
	return binary_search(neighboor + offset[x], neighboor + offset[x + 1], y);
}