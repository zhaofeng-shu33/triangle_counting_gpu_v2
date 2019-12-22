#include "gpu.h"
#include "gpu-thrust.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <thread>

#include "TrCountingGraph.h"
using namespace std;

#define NUM_THREADS 64
#define NUM_BLOCKS_GENERIC 112
#define NUM_BLOCKS_PER_MP 8

void CudaAssert(cudaError_t status, const char* code, const char* file, int l) {
  if (status == cudaSuccess) return;
  cerr << "Cuda error: " << code << ", file " << file << ", line " << l << endl;
  exit(1);
}

#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

__global__ void CalculateTrianglesSplit_v2(TrCountingGraphChunk* chunk, int deviceCount = 1, int deviceIdx = 0) {
    int from =
    gridDim.x * blockDim.x * deviceIdx +
    blockDim.x * blockIdx.x +
    threadIdx.x;
  int step = deviceCount * gridDim.x * blockDim.x;
  
  uint64_t count = 0;
  for (int k = from; k < chunk->length; k += step) {
    int i =  chunk->dev_neighbor_start_i[k]; 
    int j =  chunk->dev_neighbor_i[k]; 
    if (j == 2147483647 || chunk->dev_offset[j] < chunk->dev_split_offset[chunk->chunkid_j] || chunk->dev_offset[j] >= chunk->dev_split_offset[chunk->chunkid_j+1]) 
        continue;
    int64_t j_it = chunk->dev_offset[j]-chunk->dev_split_offset[chunk->chunkid_j];
    int64_t i_it = chunk->dev_offset[i]-chunk->dev_split_offset[chunk->chunkid_i];
    int64_t j_it_end = j_it+chunk->dev_degree[j]-1;
    int64_t i_it_end = i_it+chunk->dev_degree[i]-1;

    int a = chunk->dev_neighbor_i[i_it], b = chunk->dev_neighbor_j[j_it]; 
    while(j_it <= j_it_end && i_it <= i_it_end) {
      int d = a - b;
      if(d == 0) {
        count++;
      }
      if(d <= 0)
        a = chunk->dev_neighbor_i[++i_it]; 
      if(d >= 0)
        b = chunk->dev_neighbor_j[++j_it];
    }
  }
  chunk->dev_results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

__global__ void CalculateTriangles_v2(TrCountingGraphChunk* chunk, int n, uint64_t* results, int deviceCount = 1,
    int deviceIdx = 0) {
    int from =
    gridDim.x * blockDim.x * deviceIdx +
    blockDim.x * blockIdx.x +
    threadIdx.x;
  int step = deviceCount * gridDim.x * blockDim.x;
  
  uint64_t count = 0;
  for (int k = from; k < n; k += step) {
    int i =  chunk->dev_neighbor_start[k]; 
    int j =  chunk->dev_neighbor[k]; 
    if (j==2147483647) continue;
    int64_t j_it = chunk->dev_offset[j];
    int64_t i_it = chunk->dev_offset[i];
    int64_t j_it_end = j_it+chunk->dev_degree[j]-1;
    int64_t i_it_end = i_it+chunk->dev_degree[i]-1;

    int a = chunk->dev_neighbor[i_it], b = chunk->dev_neighbor[j_it]; 
    while(j_it <= j_it_end && i_it <= i_it_end){
      int d = a-b;
      if ( d == 0 ){
        count++;
      }
      if (d <= 0)
        a = chunk->dev_neighbor[++i_it]; 
      if (d >= 0)
        b = chunk->dev_neighbor[++j_it];
    }
  }
  results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

int NumberOfMPs() {
  int dev, val;
  CUCHECK(cudaGetDevice(&dev));
  CUCHECK(cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, dev));
  return val;
}

size_t GlobalMemory() {
  int dev;
  cudaDeviceProp prop;
  CUCHECK(cudaGetDevice(&dev));
  CUCHECK(cudaGetDeviceProperties(&prop, dev));
  return prop.totalGlobalMem;
}

uint64_t GpuForward_v2(const TrCountingGraph& TrCountingGraph) {
    const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();
    TrCountingGraphChunk chunk(TrCountingGraph, 1, TrCountingGraph.edge_num);

    uint64_t* dev_results;
    CUCHECK(cudaMalloc(&dev_results,
          NUM_BLOCKS * NUM_THREADS * sizeof(uint64_t)));
    
    cudaFuncSetCacheConfig(CalculateTriangles_v2, cudaFuncCachePreferL1);
    CalculateTriangles_v2<<<NUM_BLOCKS, NUM_THREADS>>>(chunk.dev_this, 
        TrCountingGraph.offset[TrCountingGraph.nodeid_max+1], dev_results);
    CUCHECK(cudaDeviceSynchronize());
    uint64_t result = SumResults(NUM_BLOCKS * NUM_THREADS, dev_results);
    return result;
}

int GetDevNum(){
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  return deviceCount;
}

int GetSplitNum(int num_nodes, uint64_t num_edges) {
  int DevNum = GetDevNum();
  int split_num = 0;
  int split_num_new = 0;
  for (int i = 0; i < DevNum; i++){
    CUCHECK(cudaSetDevice(i));
    uint64_t mem = (uint64_t)GlobalMemory();  // in Byte
    mem -= (uint64_t)num_nodes * 16;  // uint64_t
    split_num_new = (int)(1 + 12 * (num_edges) / mem);
    split_num = split_num > split_num_new ? split_num : split_num_new;
  }
  return split_num;
}
void calculation_thread(TrCountingGraph* TrCountingGraph, int split_num, int64_t cpu_offset,int gpu_offset_start, int gpu_offset_end, int rank, int step, int64_t* result_temp){
  CUCHECK(cudaSetDevice(rank));
  const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();
  TrCountingGraphChunk chunk(*TrCountingGraph, split_num, cpu_offset);
  int64_t result=0;
  for(int ij = gpu_offset_start+rank; ij < gpu_offset_end; ij=ij+step) {
      int i, j;
      get_i_j(split_num, ij, &i, &j);
      if(chunk.split_offset[i] >= cpu_offset)
          break;
      chunk.initChunk(i, j);
      
      CalculateTrianglesSplit_v2<<<NUM_BLOCKS, NUM_THREADS>>>(chunk.dev_this);
      CUCHECK(cudaDeviceSynchronize());
      result = result + SumResults(NUM_BLOCKS * NUM_THREADS, chunk.dev_results);
  }
  *result_temp = result;
}

uint64_t GpuForwardSplit_v2(TrCountingGraph& TrCountingGraph, 
    int split_num, int64_t cpu_offset,
    int gpu_offset_start, int gpu_offset_end) {
    
    int DevNum = GetDevNum();
    int64_t result = 0;
    int64_t* result_temp = new int64_t[DevNum];
    thread** thread_list = new thread*[DevNum];
    for(int i=0;i<DevNum;i++){
      thread_list[i] = new thread(calculation_thread,&TrCountingGraph,split_num,cpu_offset,gpu_offset_start,gpu_offset_end,i,DevNum,result_temp+i);
    }
    for(int i=0;i<DevNum;i++){
      thread_list[i]->join();
      result += result_temp[i];
    }
    return result;
}

TrCountingGraphChunk::TrCountingGraphChunk(const TrCountingGraph& g, int split_num, int64_t cpu_task){
  chunk_length_max = get_split_v2(g.offset, g.nodeid_max, split_num, split_offset);
  // device memory initialization
  const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();
  cpu_offset = cpu_task;
  CUCHECK(cudaMalloc(&dev_results, NUM_BLOCKS * NUM_THREADS * sizeof(uint64_t)));
  CUCHECK(cudaMalloc(&dev_offset, (g.nodeid_max + 2) * sizeof(int64_t)));
  CUCHECK(cudaMemcpy(
      dev_offset, g.offset, (g.nodeid_max + 2) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUCHECK(cudaMalloc(&dev_degree, (g.nodeid_max + 1) * sizeof(int)));
  CUCHECK(cudaMemcpy(
      dev_degree, g.degree, (g.nodeid_max + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUCHECK(cudaMalloc(&dev_split_offset, (split_num + 1) * sizeof(int64_t)));
  CUCHECK(cudaMemcpy(
      dev_split_offset, split_offset, (split_num + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  if(split_num==1){
    CUCHECK(cudaMalloc(&dev_neighbor, (g.edge_num) * sizeof(int)));
    CUCHECK(cudaMemcpy(
       dev_neighbor, g.neighboor, (g.edge_num) * sizeof(int), cudaMemcpyHostToDevice));
    CUCHECK(cudaMalloc(&dev_neighbor_start, (g.edge_num) * sizeof(int)));
    CUCHECK(cudaMemcpy(
       dev_neighbor_start, g.neighboor_start, (g.edge_num) * sizeof(int), cudaMemcpyHostToDevice));
  }
  else{
    CUCHECK(cudaMalloc(&dev_neighbor_i, chunk_length_max * sizeof(int)));
    CUCHECK(cudaMalloc(&dev_neighbor_start_i, chunk_length_max * sizeof(int)));
    CUCHECK(cudaMalloc(&dev_neighbor_j, chunk_length_max * sizeof(int)));
  }
  cudaFuncSetCacheConfig(CalculateTrianglesSplit_v2, cudaFuncCachePreferL1);

  Graph = &g;
  CUCHECK(cudaMalloc(&dev_this, sizeof(TrCountingGraphChunk)));
  CUCHECK(cudaMemcpy(dev_this,this,sizeof(TrCountingGraphChunk),cudaMemcpyHostToDevice));
}
void TrCountingGraphChunk::initChunk(int i, int j){
  chunkid_i = i;
  chunkid_j = j;
  length = cpu_offset>split_offset[i+1]?split_offset[i+1]-split_offset[i]:cpu_offset-split_offset[i];
  CUCHECK(cudaMemcpy(
    dev_neighbor_i, Graph->neighboor+split_offset[i], (split_offset[i+1]-split_offset[i])*sizeof(int), cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(
    dev_neighbor_start_i, Graph->neighboor_start+split_offset[i], (split_offset[i+1]-split_offset[i])*sizeof(int), cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(
    dev_neighbor_j, Graph->neighboor+split_offset[j], (split_offset[j+1]-split_offset[j])*sizeof(int), cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(dev_this,this,sizeof(TrCountingGraphChunk),cudaMemcpyHostToDevice));
}