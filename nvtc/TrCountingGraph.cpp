#include "TrCountingGraph.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <cstring>
#include <set>
#include <ctime>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#define BUFFERSIZE 8192*128
#define BATCHSIZE BUFFERSIZE/8
#define INTMAX 2147483647
#define THREADNUM 8
// R4 is an IO-Dense task, slighly more threads can make better use of cpu. 
#define THREADNUM_R4 10
#define LOCKSHARE 10

using namespace std;

void foo(){return;};
void get_max(int*u, int64_t length, int64_t from, int64_t step, int* out);
void get_degree(int*u, int64_t length, int64_t from, int64_t step, int* temp2);
void get_length(int*u, int64_t length, int64_t from, int64_t step, mutex* lock, int* _temp2, int* _temp);
void loadbatch_R3(TrCountingGraph* G,const char* file_name, int length,int from,int step);


TrCountingGraph::TrCountingGraph(const char* file_name){
	// Temporal variables
    std::ifstream fin;
	char buffer[BUFFERSIZE];
	char u_array[4], v_array[4];
	int64_t counter = 0;
	int *u, *v;
	int x, y;
	int node_max = 0;
	//uint THREADNUM = thread::hardware_concurrency();
	int* node_max_thread = new int[THREADNUM]{0};
	thread* ths[THREADNUM_R4];
	int i = 0;

	// Compute edge num by file length
	fin.open(file_name, ifstream::binary | ifstream::in);
	fin.seekg(0, fin.end);
	edge_num = fin.tellg()/8;
	fin.seekg(0, fin.beg);
	
	//Round 1, Get max id
#if VERBOSE
	printf("Round 1, Get max id");
#endif
	nodeid_max = 0;
	entire_data = new char[edge_num*8];
	fin.read(entire_data, edge_num*8);
	u = reinterpret_cast<int*>(entire_data);
	for(int i=0;i<THREADNUM;i++)
		ths[i] = new thread(get_max, u, edge_num*2, i, THREADNUM, node_max_thread+i);
	for(i=0;i<THREADNUM;i++){
		ths[i]->join();
		if(node_max_thread[i]>nodeid_max)
			nodeid_max = node_max_thread[i];
	}

	//Round 2, Get node degree, use this to decide where a edge should store
#if VERBOSE
	printf("Round 2, Get degree\n");
#endif
	int* _temp2 = new int[nodeid_max + 1]();
	for(int i=0;i<THREADNUM;i++)
		ths[i] = new thread(get_degree, u, edge_num*2, 2*i, 6*THREADNUM, _temp2);
	for(i=0;i<THREADNUM;i++){
		ths[i]->join();
	}

	//Round 3, Get offset
#if VERBOSE
	printf("Round 3, Get offset");
#endif
	mutex* lock = new mutex[nodeid_max/LOCKSHARE + 1];
	int* _temp = new int[nodeid_max + 1]();
	for(int i=0;i<THREADNUM;i++)
		ths[i] = new thread(get_length, u, edge_num*2, 2*i, 2*THREADNUM, lock, _temp2, _temp);
	for(i=0;i<THREADNUM;i++){
		ths[i]->join();
	}
	if(edge_num%2==0){
		#pragma omp parallel for
		for (int64_t i = edge_num; i <= 2*edge_num; i+=2) {
			u[i-edge_num+1] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 0; i < edge_num; i+=2) {
			u[edge_num+i/2] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 1; i < edge_num; i+=2) {
			u[edge_num/2*3+i/2] = u[i];
		}
	}else{
		#pragma omp parallel for
		for (int64_t i = edge_num+1; i <= 2*edge_num; i+=2) {
			u[i-edge_num] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 0; i < edge_num; i+=2) {
			u[edge_num+i/2] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 1; i < edge_num; i+=2) {
			u[edge_num+edge_num/2+1+i/2] = u[i];
		}
	}
	
	//delete[] entire_data;
	delete [] lock;
	degree = new int[nodeid_max + 1]();
	#pragma omp parallel for
	for (int64_t i = 0; i < nodeid_max+1; i++) {
		degree[i] =  _temp[i];
	}
	neighboor = u;
	neighboor_start = u+edge_num;
	offset = new int64_t[nodeid_max +2]();
	offset[0] = 0;
	for (int64_t i = 1; i <= nodeid_max+1; i++) {
		offset[i] = offset[i - 1] + _temp[i - 1];
	}

	//Round 4, Record neighboors
#if VERBOSE
	printf("Round 4, Record neighboors");
#endif
	int64_t batch_num = edge_num/(BATCHSIZE);
	int64_t residual = edge_num%(BATCHSIZE);
	for(int i=0;i<THREADNUM_R4;i++)
		ths[i] = new thread(loadbatch_R3, this, file_name, batch_num, i, THREADNUM_R4);
	for(i=0;i<THREADNUM_R4;i++){
		ths[i]->join();
	}
	counter = batch_num*(BATCHSIZE);
	fin.seekg(batch_num*BUFFERSIZE,fin.beg);
	fin.read(buffer, residual*8);
	u = reinterpret_cast<int*>(buffer);
	int choice, shift;
	for (int64_t i = 0; i < edge_num-counter; i++) {
		x = *(u + 2 * i);
		y = *(u + 2 * i + 1);
		if (x==y) continue;
		choice = neighboor_start[counter+i]%2;
		shift = neighboor_start[counter+i]>>1;
		if( choice==0 ){
			neighboor[offset[x] + shift] = y;
		}
		else{
			neighboor[offset[y] + shift] = x;
		}
	}

	#pragma omp parallel for
	for (int64_t i = 0; i <= nodeid_max; i++) {
		int64_t start = offset[i];
		for (int j=0; j<degree[i];j++)
			neighboor_start[start+j] = i;
	}

	sort_neighboor(_temp);

	#pragma omp parallel for
	for (int64_t i = 0; i <= nodeid_max; i++) {
		int m,n;
		if (_temp[i]>1){
			for(m=0;m<_temp[i];){
				for(n=m+1;n<_temp[i] && neighboor[offset[i]+m]==neighboor[offset[i]+n];n++){
					degree[i]--;
					neighboor[offset[i]+n] = INTMAX;
				}
				m = n;
			}
		}
	}

	sort_neighboor(_temp);
}

void TrCountingGraph::sort_neighboor(int* d) {
#pragma omp parallel for
	for (int64_t i = 0; i <= nodeid_max; i++) {
		sort(neighboor + offset[i], neighboor + offset[i] + d[i]);
	}
}

void get_max(int*u, int64_t length, int64_t from, int64_t step, int* out){
	int max = 0;
	for(int64_t i = from;i<length;i+=step){
		if(u[i]>max)
			max = u[i];
	}
	*out = max;
}
void get_degree(int*u, int64_t length, int64_t from, int64_t step, int* temp2){
	for(int64_t i = from;i<length;i+=step){
		temp2[u[i]]++;
		temp2[u[i+1]]++;
	}
}
// _temp是一个计数器，用来记录这个节点下分配了多少条边，同时对某条边就是稍后实际写入时的相对位置
void get_length(int*u, int64_t length, int64_t from, int64_t step, mutex* lock, int* _temp2, int* _temp){
	int x,y;
	for (int64_t i = from; i < length; i += step) {
		x = *(u + i);
		y = *(u + i + 1);
		if (x == y)
		    continue;
		if(_temp2[x] < _temp2[y] ) {
			lock[x/LOCKSHARE].lock();
			*(u + i) = _temp[x] << 1; // 最后一位记录要分到哪个节点下面
		    _temp[x]++;
			lock[x/LOCKSHARE].unlock();
		}
		else if (_temp2[x] > _temp2[y]) {
			lock[y/LOCKSHARE].lock();
			*(u + i) = (_temp[y] << 1) + 1;
			_temp[y]++;
			lock[y/LOCKSHARE].unlock();
		}
		else if (x < y) {
			lock[x/LOCKSHARE].lock();
			*(u + i) = _temp[x] << 1;
			_temp[x]++;
			lock[x/LOCKSHARE].unlock();
		}
		else {
			lock[y/LOCKSHARE].lock();
			*(u + i) = (_temp[y] << 1) + 1;
			_temp[y]++;
			lock[y/LOCKSHARE].unlock();
		}
	}
}

void loadbatch_R3(TrCountingGraph* G,const char* file_name, int length,int from,int step){
	std::ifstream fin;
	fin.open(file_name, ifstream::binary | ifstream::in);
	int64_t start = 0;
	char buffer[BUFFERSIZE];
	int* u;
	int x,y;
	int choice,shift;
	int64_t counter;
	for (int64_t k=from; k<length; k+=step){
		fin.seekg(k*BUFFERSIZE, fin.beg);
		fin.read(buffer, BUFFERSIZE);
		counter = k*BATCHSIZE;
		u = reinterpret_cast<int*>(buffer);
		for (int j = 0; j < BATCHSIZE; j++) {
			x = *(u + 2 * j);
			y = *(u + 2 * j + 1);
			if (x==y) continue;
			choice = G->neighboor_start[counter+j]%2;
			shift = G->neighboor_start[counter+j]>>1;
			if( choice==0 ){
				G->neighboor[G->offset[x] + shift] = y;
			}
			else{
				G->neighboor[G->offset[y] + shift] = x;
			}	
		}
	}
	return;
}

int64_t get_split_v2(int64_t* offset, int nodeid_max, int split_num, int64_t*& out){
	int64_t max_length = 0;
	out = new int64_t[split_num+1];
	out[0] = 0;
	for(int i=1;i<split_num;i++){
		int64_t target = out[i-1]+(offset[nodeid_max+1])/split_num;
		out[i] = *lower_bound(offset,offset+nodeid_max+2,target);
	}
	out[split_num] = offset[nodeid_max+1];
	for(int i=1;i<=split_num;i++){
		if(out[i]-out[i-1]>max_length)
			max_length = out[i]-out[i-1];
	}
	return max_length;
}
void cpu_counting_edge_first_v2(TrCountingGraph* g, int64_t offset_start, int64_t* out){
    int64_t sum=0;
    int iit = 0;
    int jit = 0;
    int d = 0;
    int i,j;
    #pragma omp parallel for schedule(dynamic,1024) reduction(+:sum) private(iit,jit,d,i,j)
    for (int64_t k=offset_start;k<g->offset[g->nodeid_max+1];k++){
        i = g->neighboor_start[k];
        j = g->neighboor[k];
        if(j==INTMAX)
        continue;
        iit = 0;
        jit = 0;
            while(iit<g->degree[i] && jit<g->degree[j]){
                d = g->neighboor[g->offset[i]+iit]-g->neighboor[g->offset[j]+jit];
                if(d==0){
                    sum++;
                    iit++;
                    jit++;
                }
                if(d<0){
                    iit++;
                }
                if(d>0){
                    jit++;
                }
            }
        }
    *out = sum;
	cout<<"CPU Done."<<endl;
#if VERBOSE
	cout<<"CPU Done."<<endl;
#endif
}
