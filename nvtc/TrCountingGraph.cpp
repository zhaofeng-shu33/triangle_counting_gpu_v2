#include "TrCountingGraph.h"
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <pthread.h>
#if VERBOSE
#include <time.h>
#endif
#define BUFFERSIZE (8192*128)
#define BATCHSIZE (BUFFERSIZE/8)
#define INTMAX 2147483647
#define THREADNUM 8
// R4 is an IO-Dense task, slightly more threads can make better use of cpu. 
#define THREADNUM_R4 10
#define LOCKSHARE 10

struct GET_LENGTH_ARGS {
	int* u;
	int64_t length;
	int64_t from;
	int64_t step;
	pthread_mutex_t* lock;
	int* degree_estimation;
	int* pointer;
};

struct BATCH_R4_ARGS {
	TrCountingGraph* G;
	const char* file_name;
	int length;
	int from;
	int step;
};

using namespace std;

void* get_length(void* args);
void* loadbatch_R4(void* args);
void construct_trCountingGraph(TrCountingGraph* tr_graph, const char* file_name);

TrCountingGraph::TrCountingGraph(const char* file_name) {
	construct_trCountingGraph(this, file_name);
}

int64_t get_edge_num(FILE* pFile) {
	fseek(pFile, 0, SEEK_END);
	int64_t size = ftell(pFile);
	rewind(pFile);
	return size / 8;
}

int get_max_id(int* data, int64_t len) {
	int max_node_id = 0;
	#pragma omp parallel for reduction(max:max_node_id)
	for (int64_t i = 0; i < len; i += 1) {
		if (max_node_id < data[i])
			max_node_id = data[i];
	}
	return max_node_id;
}

void construct_trCountingGraph(TrCountingGraph* tr_graph, const char* file_name) {
	// Temporal variables
	char buffer[BUFFERSIZE];
	int64_t counter = 0;
	int *u, *v;
	int x, y;
	int node_max = 0;
	pthread_t ths[THREADNUM_R4];
	// Compute edge num by file length
	FILE* pFile = fopen(file_name, "rb");
	tr_graph->edge_num = get_edge_num(pFile);
	
	//Round 1, Get max id
#if VERBOSE
    int rank = 0;
#if MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	printf("Rank %d: Round 1, Get max id\n", rank);
    time_t start_t = time(NULL);
#endif
	tr_graph->nodeid_max = 0;
	tr_graph->entire_data = (char*)malloc(sizeof(char) * tr_graph->edge_num * 8);
	fread(tr_graph->entire_data, 2 * sizeof(int), tr_graph->edge_num, pFile);
	u = (int*)(tr_graph->entire_data);
	tr_graph->nodeid_max = get_max_id(u, tr_graph->edge_num * 2);

	//Round 2, Get node degree, use this to decide where a edge should store
#if VERBOSE
    time_t end_t = time(NULL);
    int duration_t = end_t - start_t;
    printf("Rank %d: Round 1 used %d seconds; Round 2, Get degree\n", rank, duration_t);
    start_t = end_t;
#endif
	int* degree_estimation;
	degree_estimation = (int*)malloc(sizeof(int) * (tr_graph->nodeid_max + 1));
	memset(degree_estimation, 0, sizeof(int) * (tr_graph->nodeid_max + 1));
#ifndef USEMPI
	#pragma omp parallel for
#endif
	for (int64_t i = 0; i < tr_graph->edge_num * 2; i += 6) {
		// This is only a rough estimation, ignoring racing in multi-thread
		// heuristically it is actually an estimation of degree
		// why not using exact degree?
		// Reason 1: the input data is polluted, exact degree is impossible 
		// at this step;
		// Reason 2: empirical experiments show that the triangle counting
		// time is similar if using lock and i += 2 at this step.
		degree_estimation[u[i]]++;
		degree_estimation[u[i + 1]]++;
	}

	//Round 3, Get offset
#if VERBOSE
    end_t = time(NULL);
    duration_t = end_t - start_t;
    start_t = end_t;
    printf("Rank %d: Round 2 used %d seconds; Round 3, Get offset\n", rank, duration_t);
#endif
	int num_of_thread_locks = tr_graph->nodeid_max/LOCKSHARE + 1;
	pthread_mutex_t* lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t) * num_of_thread_locks);
	for(int i = 0; i < num_of_thread_locks; i++) {
		// we do not use std::mutex as pthread_mutex is faster
		pthread_mutex_init(&lock[i], NULL);
	}
	int* pointer; // exact degree
	pointer = (int*)malloc(sizeof(int) * (tr_graph->nodeid_max + 1));
	memset(pointer, 0, sizeof(int) * (tr_graph->nodeid_max + 1));
	struct GET_LENGTH_ARGS gen_length_args_array[THREADNUM];	
	for (int i = 0; i < THREADNUM; i++) {
		gen_length_args_array[i] = {u, tr_graph->edge_num * 2, 2 * i,
			2 * THREADNUM, lock, degree_estimation, pointer};
		// pthread_create only supports void* fun(void*),
		// therefore constructing struct to pass multiple
		// variable is necessary.
		pthread_create(&ths[i], NULL, get_length, (void *)&gen_length_args_array[i]);
	}

	for (int i = 0; i < THREADNUM; i++) {
		pthread_join(ths[i], NULL);
	}
    // (1,2,3,4,5,6) -> (1,3,5,2,4,6)
	if (tr_graph->edge_num % 2==0) {
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
	} else {
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
	
	for(int i = 0; i < num_of_thread_locks; i++) {
		pthread_mutex_destroy(&lock[i]);
	}
	free(lock);
	free(degree_estimation);
	tr_graph->degree = pointer;
	tr_graph->neighboor = u;
	tr_graph->neighboor_start = u + tr_graph->edge_num;
	tr_graph->offset = (int64_t*) malloc(sizeof(int64_t) * (tr_graph->nodeid_max + 2));
	memset(tr_graph->offset, 0, sizeof(int64_t) * (tr_graph->nodeid_max + 2));
	tr_graph->offset[0] = 0;
	for (int64_t i = 1; i <= tr_graph->nodeid_max + 1; i++) {
		tr_graph->offset[i] = tr_graph->offset[i - 1] + pointer[i - 1];
	}  // tr_graph->offset[-1] save the edge num
	// Round 4, Record neighboors
#if VERBOSE
	end_t = time(NULL);
    duration_t = end_t - start_t;
    start_t = end_t;
    printf("Rank %d: Round 3 used %d seconds; Round 4, Record neighboors\n", rank, duration_t);
#endif
    // batch_num must be int64_t
	int64_t batch_num = tr_graph->edge_num / BATCHSIZE;
	int64_t residual = tr_graph->edge_num % BATCHSIZE;
	struct BATCH_R4_ARGS batch_r4_args_array[THREADNUM_R4];
	for (int i = 0; i < THREADNUM_R4; i++){
		batch_r4_args_array[i] = {tr_graph, file_name, batch_num, i, THREADNUM_R4};
		pthread_create(&ths[i], NULL, loadbatch_R4, (void *)&batch_r4_args_array[i]);
	}
	for (int i = 0; i < THREADNUM_R4; i++) {
		pthread_join(ths[i], NULL);
	}
	counter = batch_num * BATCHSIZE;
	fseek(pFile, batch_num * BUFFERSIZE, SEEK_SET);
	fread(buffer, 2 * sizeof(int), residual, pFile);

	u = (int*)(buffer);
	int choice, shift;
	for (int64_t i = 0; i < tr_graph->edge_num-counter; i++) {
		x = *(u + 2 * i);
		y = *(u + 2 * i + 1);
		if(x == y) continue;
		choice = tr_graph->neighboor_start[counter + i] % 2;
		shift = tr_graph->neighboor_start[counter + i] >> 1;
		if(choice==0) {
			tr_graph->neighboor[tr_graph->offset[x] + shift] = y;
		} else {
			tr_graph->neighboor[tr_graph->offset[y] + shift] = x;
		}
	}

	#pragma omp parallel for
	for (int64_t i = 0; i <= tr_graph->nodeid_max; i++) {
		int64_t start = tr_graph->offset[i];
		for (int j = 0; j < tr_graph->degree[i]; j++)
			tr_graph->neighboor_start[start + j] = i;
	}

	sort_neighboor(tr_graph);

	#pragma omp parallel for
	for (int64_t i = 0; i <= tr_graph->nodeid_max; i++) {
		int m, n, degree_inner = 0;
		if (pointer[i] > 1) {
			for (m = 0; m < pointer[i];) {
				for (n = m + 1; n < pointer[i] &&
					tr_graph->neighboor[tr_graph->offset[i] + m] ==
					tr_graph->neighboor[tr_graph->offset[i] + n]; n++) {
					degree_inner++;
					tr_graph->neighboor[tr_graph->offset[i] + n] = INTMAX;
				}
				m = n;
			}
			tr_graph->degree[i] -= degree_inner;
		}
	}
	sort_neighboor(tr_graph);
#if VERBOSE
 	end_t = time(NULL);
    duration_t = end_t - start_t;
    printf("Rank %d: Round 4 used %d seconds; Read data done\n", rank, duration_t);
#endif
}

void sort_neighboor(TrCountingGraph* g) {
	#pragma omp parallel for
	for (int64_t i = 0; i <= g->nodeid_max; i++) {
		sort(g->neighboor + g->offset[i], g->neighboor + g->offset[i + 1]);
	}
}

// pointer 是一个计数器，用来记录这个节点下分配了多少条边，同时对某条边就是稍后实际写入时的相对位置
// degree_estimation 是度的估计
void* get_length(void* args) {
	struct GET_LENGTH_ARGS* gen_len_args = (struct GET_LENGTH_ARGS*) args;
	int* u = gen_len_args->u;
	int64_t length = gen_len_args->length;
	int64_t from = gen_len_args->from;
	int64_t step = gen_len_args->step;
	pthread_mutex_t* lock = gen_len_args->lock;
	int* degree_estimation = gen_len_args->degree_estimation;
	int* pointer = gen_len_args->pointer;
	int x,y;
	for (int64_t i = from; i < length; i += step) {
		x = *(u + i);
		y = *(u + i + 1);
		if (x == y)
		    continue;
		if(degree_estimation[x] < degree_estimation[y] ) {
			pthread_mutex_lock(&lock[x/LOCKSHARE]);
			*(u + i) = pointer[x] << 1; // 最后一位记录要分到哪个节点下面
		    pointer[x]++;
			pthread_mutex_unlock(&lock[x/LOCKSHARE]);
		}
		else if (degree_estimation[x] > degree_estimation[y]) {
			pthread_mutex_lock(&lock[y/LOCKSHARE]);
			*(u + i) = (pointer[y] << 1) + 1;
			pointer[y]++;
			pthread_mutex_unlock(&lock[y/LOCKSHARE]);
		}
		else if (x < y) {
			pthread_mutex_lock(&lock[x/LOCKSHARE]);
			*(u + i) = pointer[x] << 1;
			pointer[x]++;
			pthread_mutex_unlock(&lock[x/LOCKSHARE]);
		}
		else {
			pthread_mutex_lock(&lock[y/LOCKSHARE]);
			*(u + i) = (pointer[y] << 1) + 1;
			pointer[y]++;
			pthread_mutex_unlock(&lock[y/LOCKSHARE]);
		}
	}
	return NULL;
}

void* loadbatch_R4(void* args) {
	struct BATCH_R4_ARGS* batch_r4_args = (struct BATCH_R4_ARGS*) args;
	TrCountingGraph* G = batch_r4_args->G;
	const char* file_name = batch_r4_args->file_name;
	int length = batch_r4_args->length;
	int from = batch_r4_args->from;
	int step = batch_r4_args->step;
	FILE* pFile = fopen(file_name, "rb");
	int64_t start = 0;
	char* buffer = (char*)malloc(BUFFERSIZE);
	int* u;
	int x,y;
	int choice,shift;
	int64_t counter;
	for (int64_t k = from; k < length; k += step) {
		fseek(pFile, k * BUFFERSIZE, SEEK_SET);
		fread(buffer, 1, BUFFERSIZE, pFile);
		counter = k * BATCHSIZE;
		u = (int*)(buffer);
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
	free(buffer);
	return NULL;
}

int64_t get_split_v2(int64_t* offset, int nodeid_max, int split_num, int64_t*& out) {
	int64_t max_length = 0;
	out = (int64_t*)malloc(sizeof(int64_t) * (split_num + 1));
	memset(out, 0, sizeof(int64_t) * (split_num + 1));
	out[0] = 0;
	for(int i = 1; i < split_num; i++){
		int64_t target = out[i - 1] + (offset[nodeid_max + 1]) / split_num;
		out[i] = *lower_bound(offset, offset + nodeid_max + 2, target);
	}
	out[split_num] = offset[nodeid_max + 1];
	for(int i = 1; i <= split_num; i++){
		if(out[i] - out[i-1] > max_length)
			max_length = out[i] - out[i - 1];
	}
	return max_length;
}

void cpu_counting_edge_first_v2(TrCountingGraph* g, int64_t offset_start, 
    int64_t offset_end, int64_t* out) {
    int64_t sum = 0;
    int iit = 0;
    int jit = 0;
    int d = 0;
    int i,j;
    #pragma omp parallel for schedule(dynamic,1024) reduction(+:sum) private(iit,jit,d,i,j)
    for (int64_t k = offset_start; k < offset_end; k++) {
        i = g->neighboor_start[k];
        j = g->neighboor[k];
        if(j == INTMAX)
            continue;
        iit = 0;
        jit = 0;
        while(iit < g->degree[i] && jit < g->degree[j]) {
            d = g->neighboor[g->offset[i] + iit] - g->neighboor[g->offset[j] + jit];
            if(d == 0) {
                sum++;
                iit++;
                jit++;
            }
            if(d < 0) {
                iit++;
            }
            if(d > 0) {
                jit++;
            }
        }
    }
    *out = sum;
}

//! decode (i, j) pair
void get_i_j(int n, int ij, int* i, int* j) {
    *i = ij / n;
    *j = ij % n;
}
