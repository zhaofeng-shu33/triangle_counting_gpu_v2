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
	cout << "Round 1, Get max id" << endl;
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

	// Call for mem
	offset = new int64_t[nodeid_max +2];
	degree = new int[nodeid_max + 1];
	length = new int[nodeid_max + 1];

	//Round 2, Get node degree
	cout << "Round 2, Get node degree" << endl;
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
			if(x<y)
				length[x]++;
			else
				length[y]++;
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
		if(x<y)
			length[x]++;
		else
			length[y]++;
	}
	
	offset[0] = 0;
	for (int64_t i = 1; i <= nodeid_max+1; i++) {
		offset[i] = offset[i - 1] + length[i - 1];
	}

	// Call for mem
	neighboor = new int[edge_num];
	int* _pointer = new int[nodeid_max + 1];

	//Round 3, Record neighboors
	cout << "Round 3, Record neighboors" << endl;
	fin.seekg(0, fin.beg);
	counter = 0;
	while (counter + BATCHSIZE < edge_num) {
		fin.read(buffer, BUFFERSIZE);
		u = reinterpret_cast<int*>(buffer);
		for (int j = 0; j < BATCHSIZE; j++) {
			x = *(u + 2 * j);
			y = *(u + 2 * j + 1);
			if (x!=y && !inner_arc_exist(x,y,_pointer)){
				if(x<y)
					neighboor[offset[x] + _pointer[x]++] = y;
				else
					neighboor[offset[y] + _pointer[y]++] = x;
			}
			else
			{
				degree[x]--;
				degree[y]--;
				if(x<y)
					length[x]--;
				else
					length[y]--;
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
		if (x!=y && !inner_arc_exist(x,y,_pointer)){
				if(x<y)
					neighboor[offset[x] + _pointer[x]++] = y;
				else
					neighboor[offset[y] + _pointer[y]++] = x;
			}
			else
			{
				degree[x]--;
				degree[y]--;
				if(x<y)
					length[x]--;
				else
					length[y]--;
			}
	}

	sort_neighboor();

	cout<<"Data Done."<<endl;
	cout<<"node max: "<<nodeid_max<<endl;
	cout<<"edge_num: "<<edge_num<<endl;
	cout<<"offset: ";
	for (int i=0;i<=nodeid_max;i++){
		cout<<offset[i]<<" ";
	}
	cout<<endl;
	cout<<"length: ";
	for (int i=0;i<=nodeid_max;i++){
		cout<<length[i]<<" ";
	}
	cout<<endl;
	cout<<"neighbor: ";
	for (int i=0;i<edge_num;i++){
		cout<<neighboor[i]<<" ";
	}
	cout<<endl;
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

bool MyGraph::inner_arc_exist(int u, int v, int* d) {
	for (int i = 0; i < d[u]; i++) {
		if (neighboor[offset[u] + i] == v) {
			return true;
		}
	}
	for (int i = 0; i < d[v]; i++) {
		if (neighboor[offset[v] + i] == u) {
			return true;
		}
	}
	return false;
}

void MyGraph::sort_neighboor() {
#pragma omp parallel for
	for (int64_t i = 0; i <= nodeid_max; i++) {
		sort(neighboor + offset[i], neighboor + offset[i] + length[i]);
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
	return binary_search(neighboor + offset[x], neighboor + offset[x] + degree[x], y);
}
