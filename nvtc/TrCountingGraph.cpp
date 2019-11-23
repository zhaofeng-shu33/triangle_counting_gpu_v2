#include "TrCountingGraph.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <thread>
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
void construct_trCountingGraph(TrCountingGraph* tr_graph, const char* file_name);
int64_t get_edge_num(FILE* file);

TrCountingGraph::TrCountingGraph(const char* file_name) {
	construct_trCountingGraph(this, file_name);
}

int64_t get_edge_num(FILE* pFile) {
	fseek(pFile, 0, SEEK_END);
	int64_t size = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);
	return size / 8;
}

void construct_trCountingGraph(TrCountingGraph* tr_graph, const char* file_name){
	// Temporal variables
    ifstream fin;
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
	tr_graph->edge_num = fin.tellg()/8;
	fin.seekg(0, fin.beg);
	
	//Round 1, Get max id
#if VERBOSE
	printf("Round 1, Get max id");
#endif
	tr_graph->nodeid_max = 0;
	tr_graph->entire_data = new char[tr_graph->edge_num * 8];
	fin.read(tr_graph->entire_data, tr_graph->edge_num * 8);
	u = reinterpret_cast<int*>(tr_graph->entire_data);
	for (int i = 0;i < THREADNUM; i++)
		ths[i] = new thread(get_max, u, tr_graph->edge_num * 2, i, THREADNUM, node_max_thread+i);
	for (i = 0; i < THREADNUM; i++) {
		ths[i]->join();
		if (node_max_thread[i] > tr_graph->nodeid_max)
			tr_graph->nodeid_max = node_max_thread[i];
	}

	//Round 2, Get node degree, use this to decide where a edge should store
#if VERBOSE
	printf("Round 2, Get degree\n");
#endif
	int* _temp2 = new int[tr_graph->nodeid_max + 1]();
	for(int i=0;i<THREADNUM;i++)
		ths[i] = new thread(get_degree, u, tr_graph->edge_num * 2, 2 * i, 6*THREADNUM, _temp2);
	for(i=0;i<THREADNUM;i++){
		ths[i]->join();
	}

	//Round 3, Get offset
#if VERBOSE
	printf("Round 3, Get offset");
#endif
	mutex* lock = new mutex[tr_graph->nodeid_max/LOCKSHARE + 1];
	int* _temp = new int[tr_graph->nodeid_max + 1]();
	for(int i=0;i<THREADNUM;i++)
		ths[i] = new thread(get_length, u, tr_graph->edge_num * 2, 2*i, 2*THREADNUM, lock, _temp2, _temp);
	for(i=0;i<THREADNUM;i++){
		ths[i]->join();
	}
	if(tr_graph->edge_num % 2==0){
		#pragma omp parallel for
		for (int64_t i = tr_graph->edge_num; i <= 2 * tr_graph->edge_num; i+=2) {
			u[i - tr_graph->edge_num + 1] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 0; i < tr_graph->edge_num; i+=2) {
			u[tr_graph->edge_num + i/2] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 1; i < tr_graph->edge_num; i+=2) {
			u[tr_graph->edge_num/2*3+i/2] = u[i];
		}
	}else{
		#pragma omp parallel for
		for (int64_t i = tr_graph->edge_num+1; i <= 2 * tr_graph->edge_num; i+=2) {
			u[i - tr_graph->edge_num] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 0; i < tr_graph->edge_num; i+=2) {
			u[tr_graph->edge_num+i/2] = u[i];
		}
		#pragma omp parallel for
		for (int64_t i = 1; i < tr_graph->edge_num; i+=2) {
			u[tr_graph->edge_num + tr_graph->edge_num/2+1+i/2] = u[i];
		}
	}
	
	//delete[] entire_data;
	delete [] lock;
	tr_graph->degree = new int[tr_graph->nodeid_max + 1]();
	#pragma omp parallel for
	for (int64_t i = 0; i < tr_graph->nodeid_max+1; i++) {
		tr_graph->degree[i] =  _temp[i];
	}
	tr_graph->neighboor = u;
	tr_graph->neighboor_start = u + tr_graph->edge_num;
	tr_graph->offset = new int64_t[tr_graph->nodeid_max +2]();
	tr_graph->offset[0] = 0;
	for (int64_t i = 1; i <= tr_graph->nodeid_max + 1; i++) {
		tr_graph->offset[i] = tr_graph->offset[i - 1] + _temp[i - 1];
	}

	//Round 4, Record neighboors
#if VERBOSE
	printf("Round 4, Record neighboors");
#endif
	int64_t batch_num = tr_graph->edge_num/(BATCHSIZE);
	int64_t residual = tr_graph->edge_num%(BATCHSIZE);
	for(int i=0;i<THREADNUM_R4;i++)
		ths[i] = new thread(loadbatch_R3, tr_graph, file_name, batch_num, i, THREADNUM_R4);
	for(i=0;i<THREADNUM_R4;i++){
		ths[i]->join();
	}
	counter = batch_num*(BATCHSIZE);
	fin.seekg(batch_num*BUFFERSIZE,fin.beg);
	fin.read(buffer, residual*8);
	u = reinterpret_cast<int*>(buffer);
	int choice, shift;
	for (int64_t i = 0; i < tr_graph->edge_num-counter; i++) {
		x = *(u + 2 * i);
		y = *(u + 2 * i + 1);
		if (x==y) continue;
		choice = tr_graph->neighboor_start[counter+i]%2;
		shift = tr_graph->neighboor_start[counter+i]>>1;
		if( choice==0 ){
			tr_graph->neighboor[tr_graph->offset[x] + shift] = y;
		}
		else{
			tr_graph->neighboor[tr_graph->offset[y] + shift] = x;
		}
	}

	#pragma omp parallel for
	for (int64_t i = 0; i <= tr_graph->nodeid_max; i++) {
		int64_t start = tr_graph->offset[i];
		for (int j=0; j < tr_graph->degree[i];j++)
			tr_graph->neighboor_start[start+j] = i;
	}

	sort_neighboor(tr_graph, _temp);

	#pragma omp parallel for
	for (int64_t i = 0; i <= tr_graph->nodeid_max; i++) {
		int m,n;
		if (_temp[i] > 1) {
			for (m = 0; m < _temp[i];) {
				for(n=m+1;n<_temp[i] && tr_graph->neighboor[tr_graph->offset[i]+m] == tr_graph->neighboor[tr_graph->offset[i]+n];n++){
					tr_graph->degree[i]--;
					tr_graph->neighboor[tr_graph->offset[i]+n] = INTMAX;
				}
				m = n;
			}
		}
	}
	sort_neighboor(tr_graph, _temp);
}

void sort_neighboor(TrCountingGraph* g, int* d) {
#pragma omp parallel for
	for (int64_t i = 0; i <= g->nodeid_max; i++) {
		sort(g->neighboor + g->offset[i], g->neighboor + g->offset[i] + d[i]);
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
	ifstream fin;
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
}
