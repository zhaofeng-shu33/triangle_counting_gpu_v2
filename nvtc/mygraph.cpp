#include "MyGraph.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <set>
#include <ctime>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <thread>
#include <mutex>
#define BUFFERSIZE 8192*64
#define BATCHSIZE BUFFERSIZE/8
#define INTMAX 2147483647
#define THREADNUM 8
#define LOCKSHARE 10

using namespace std;

void foo(){return;};
void loadbatch_R3(MyGraph* G,std::ifstream* fin, int64_t counter, int* _temp2, bool* state);
void get_max(int*u, int64_t length, int64_t from, int64_t step, int* out);
void get_degree(int*u, int64_t length, int64_t from, int64_t step, int* temp2);
void get_length(int*u, int64_t length, int64_t from, int64_t step, mutex* lock, int* _temp2, int* _temp);

MyGraph::MyGraph(const char* file_name){
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
	thread* ths[THREADNUM];
	bool* thread_state = new bool[THREADNUM]{false};
	int i = 0;

	// Compute edge num by file length
	fin.open(file_name, ifstream::binary | ifstream::in);
	fin.seekg(0, fin.end);
	edge_num = fin.tellg()/8;
	fin.seekg(0, fin.beg);
	
	//Round 1, Get max id
#if VERBOSE
	cout << "Round 1, Get max id" << endl;
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
	cout << "Round 2, Get degree" << endl;
#endif
	int* _temp2 = new int[nodeid_max + 1]();
	for(int i=0;i<THREADNUM;i++)
		ths[i] = new thread(get_degree, u, edge_num*2, 2*i, 2*THREADNUM, _temp2);
	for(i=0;i<THREADNUM;i++){
		ths[i]->join();
	}

	//Round 3, Get offset
#if VERBOSE
	cout << "Round 3, Get offset" << endl;
#endif
	mutex* lock = new mutex[nodeid_max/LOCKSHARE + 1];
	int* _temp = new int[nodeid_max + 1];
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
	}else{
		#pragma omp parallel for
		for (int64_t i = edge_num+1; i <= 2*edge_num; i+=2) {
			u[i-edge_num] = u[i];
		}
	}
	
	//delete[] entire_data;
	delete [] lock;
	degree = new int[nodeid_max + 1]();
	#pragma omp parallel for
	for (int64_t i = 0; i < nodeid_max+1; i++) {
		degree[i] =  _temp[i];
	}
	neighboor = u+edge_num;
	neighboor_start = u;
	offset = new int64_t[nodeid_max +2]();
	offset[0] = 0;
	for (int64_t i = 1; i <= nodeid_max+1; i++) {
		offset[i] = offset[i - 1] + _temp[i - 1];
	}

	//Round 4, Record neighboors
#if VERBOSE
	cout << "Round 4, Record neighboors" << endl;
#endif
	fin.seekg(0, fin.beg);
	counter = 0;
	i = 0;
	while (counter + BATCHSIZE < edge_num ) {
		if(!thread_state[i]){
			thread_state[i] = true;
			if (ths[i]->joinable())
				ths[i]->join();
			ths[i]->~thread();
			ths[i] = new thread(loadbatch_R3,this,&fin,counter,_temp2,thread_state+i);
			counter = counter + BATCHSIZE;
		}
		i = (i+1)%THREADNUM;	
	}
	for(i=0;i<THREADNUM;i++){
		if(ths[i]->joinable())
			ths[i]->join();
	}
	fin.read(buffer, (edge_num-counter)*8);
	u = reinterpret_cast<int*>(buffer);
	int64_t shift;
	for (int64_t i = 0; i < edge_num-counter; i++) {
		if((counter+i)*2>=edge_num){
			if(edge_num%2==0)
				shift = ((counter+i)*2)-edge_num+1;
			else
				shift = ((counter+i)*2)-edge_num;
		}else{
			shift = (counter+i)*2;
		}
		x = *(u + 2 * i);
		y = *(u + 2 * i + 1);
		if( x!=y && (_temp2[x]<_temp2[y] || (_temp2[x]==_temp2[y] && x<y) ) )
			neighboor[offset[x] + neighboor_start[shift]] = y;
		if( x!=y && (_temp2[x]>_temp2[y] || (_temp2[x]==_temp2[y] && x>y) ) )
			neighboor[offset[y] + neighboor_start[shift]] = x;
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

void MyGraph::sort_neighboor(int* d) {
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
void get_length(int*u, int64_t length, int64_t from, int64_t step, mutex* lock, int* _temp2, int* _temp){
	int x,y;
	for(int64_t i = from;i<length;i+=step){
		x = *(u + i);
		y = *(u + i + 1);
		if( x!=y && (_temp2[x]<_temp2[y] || (_temp2[x]==_temp2[y] && x<y) ) ){
			lock[x/LOCKSHARE].lock();
			*(u + i) = _temp[x]++;
			lock[x/LOCKSHARE].unlock();
		}
		if( x!=y && (_temp2[x]>_temp2[y] || (_temp2[x]==_temp2[y] && x>y) ) ){
			lock[y/LOCKSHARE].lock();
			*(u + i) = _temp[y]++;
			lock[y/LOCKSHARE].unlock();
		}	
	}
}


void loadbatch_R3(MyGraph* G,std::ifstream* fin, int64_t counter, int* _temp2, bool* state){
	char buffer[BUFFERSIZE];
	G->fin_lock.lock();
	counter = fin->tellg()/8;
	fin->read(buffer, BUFFERSIZE);
	G->fin_lock.unlock();
	int* u = reinterpret_cast<int*>(buffer);
	int x,y;
	int64_t shift;
	for (int j = 0; j < BATCHSIZE; j++) {
		if((counter+j)*2>=G->edge_num){
			if(G->edge_num%2==0)
				shift = ((counter+j)*2)-G->edge_num+1;
			else
				shift = ((counter+j)*2)-G->edge_num;
		}else{
			shift = (counter+j)*2;
		}	
		x = *(u + 2 * j);
		y = *(u + 2 * j + 1);
		if( x!=y && (_temp2[x]<_temp2[y] || (_temp2[x]==_temp2[y] && x<y) ) ){

			G->neighboor[G->offset[x] + G->neighboor_start[shift]] = y;

		}
		if( x!=y && (_temp2[x]>_temp2[y] || (_temp2[x]==_temp2[y] && x>y) ) ){

			G->neighboor[G->offset[y] + G->neighboor_start[shift]] = x;

		}	
	}
	*state = false;
	return;
}
int64_t get_split_v2(int64_t* offset, int nodeid_max, int split_num, int64_t cpu_offset, int64_t*& out){
	int64_t max_length = 0;
	out = new int64_t[split_num+1];
	out[0] = cpu_offset;
	for(int i=1;i<split_num;i++){
		int64_t target = out[i-1]+(offset[nodeid_max+1]-cpu_offset)/split_num;
		out[i] = *lower_bound(offset,offset+nodeid_max+2,target);
	}
	out[split_num] = offset[nodeid_max+1];
	for(int i=1;i<=split_num;i++){
		if(out[i]-out[i-1]>max_length)
			max_length = out[i]-out[i-1];
	}
	return max_length;
}
void cpu_counting_edge_first_v2(MyGraph* g, int64_t cpu_offset, int64_t* out){
    int64_t sum=0;
    int iit = 0;
    int jit = 0;
    int d = 0;
    int i,j;
    #pragma omp parallel for schedule(dynamic,1024) reduction(+:sum) private(iit,jit,d,i,j)
    for (int64_t k=0;k<cpu_offset;k++){
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
#if VERBOSE
	cout<<"CPU Done."<<endl;
#endif
}
