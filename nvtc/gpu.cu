#include "gpu.h"

#include "gpu-thrust.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "TrCountingGraph.h"
using namespace std;

#define NUM_THREADS 64
#define NUM_BLOCKS_GENERIC 112
#define NUM_BLOCKS_PER_MP 8

template<bool ZIPPED>
__global__ void CalculateNodePointers(int n, int m, int* edges, int* nodes) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i <= m; i += step) {
    int prev = i > 0 ? edges[ZIPPED ? (2 * (i - 1) + 1) : (m + i - 1)] : -1;
    int next = i < m ? edges[ZIPPED ? (2 * i + 1) : (m + i)] : n;
    for (int j = prev + 1; j <= next; ++j)
      nodes[j] = i;
  }
}

__global__ void CalculateFlags(int m, int* edges, int* nodes, bool* flags) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < m; i += step) {
    int a = edges[2 * i];
    int b = edges[2 * i + 1];
    int deg_a = nodes[a + 1] - nodes[a];
    int deg_b = nodes[b + 1] - nodes[b];
    flags[i] = (deg_a < deg_b) || (deg_a == deg_b && a < b);
  }
}

__global__ void UnzipEdges(int m, int* edges, int* unzipped_edges) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < m; i += step) {
    unzipped_edges[i] = edges[2 * i];
    unzipped_edges[m + i] = edges[2 * i + 1];
  }
}
__global__ void CalculateTrianglesSplit_v2(int n,const int64_t* __restrict__ dev_offset,
    const int* __restrict__ dev_length,const int* __restrict__ dev_neighbor_i,
    const int* __restrict__ dev_neighbor_start_i, const int* __restrict__ dev_neighbor_j,
    const int64_t* __restrict__ dev_split_offset,
    int chunkid_i, int chunkid_j, uint64_t* results, int deviceCount = 1, int deviceIdx = 0) {
    int from =
    gridDim.x * blockDim.x * deviceIdx +
    blockDim.x * blockIdx.x +
    threadIdx.x;
  int step = deviceCount * gridDim.x * blockDim.x;
  
  uint64_t count = 0;
  for (int k = from; k < n; k += step) {
    int i =  dev_neighbor_start_i[k]; 
    int j =  dev_neighbor_i[k]; 
    if (j == 2147483647 || dev_offset[j] < dev_split_offset[chunkid_j] || dev_offset[j] >= dev_split_offset[chunkid_j+1]) 
        continue;
    int64_t j_it = dev_offset[j]-dev_split_offset[chunkid_j];
    int64_t i_it = dev_offset[i]-dev_split_offset[chunkid_i];
    int64_t j_it_end = j_it+dev_length[j]-1;
    int64_t i_it_end = i_it+dev_length[i]-1;

    int a = dev_neighbor_i[i_it], b = dev_neighbor_j[j_it]; 
    while(j_it <= j_it_end && i_it <= i_it_end) {
      int d = a - b;
      if(d == 0) {
        count++;
      }
      if(d <= 0)
        a = dev_neighbor_i[++i_it]; 
      if(d >= 0)
        b = dev_neighbor_j[++j_it];
    }
  }
  results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

__global__ void CalculateTriangles_v2(int n,const int* __restrict__ dev_neighbor,
    const int64_t* __restrict__ dev_offset,const int* __restrict__ dev_length,
    const int* __restrict__ dev_neighbor_start, uint64_t* results,int deviceCount = 1,
    int deviceIdx = 0) {
    int from =
    gridDim.x * blockDim.x * deviceIdx +
    blockDim.x * blockIdx.x +
    threadIdx.x;
  int step = deviceCount * gridDim.x * blockDim.x;
  
  uint64_t count = 0;
  for (int k = from; k < n; k += step) {
    int i =  dev_neighbor_start[k]; 
    int j =  dev_neighbor[k]; 
    if (j==2147483647) continue;
    int64_t j_it = dev_offset[j];
    int64_t i_it = dev_offset[i];
    int64_t j_it_end = j_it+dev_length[j]-1;
    int64_t i_it_end = i_it+dev_length[i]-1;

    int a = dev_neighbor[i_it], b = dev_neighbor[j_it]; 
    while(j_it <= j_it_end && i_it <= i_it_end){
      int d = a-b;
      if ( d == 0 ){
        count++;
      }
      if (d <= 0)
        a = dev_neighbor[++i_it]; 
      if (d >= 0)
        b = dev_neighbor[++j_it];
    }
  }
  results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

__global__ void CalculateTriangles(
    int m, const int* __restrict__ edges, const int* __restrict__ nodes,
    uint64_t* results, int deviceCount = 1, int deviceIdx = 0) {
  int from =
    gridDim.x * blockDim.x * deviceIdx +
    blockDim.x * blockIdx.x +
    threadIdx.x;
  int step = deviceCount * gridDim.x * blockDim.x;
  uint64_t count = 0;

  for (int i = from; i < m; i += step) {
    int u = edges[i], v = edges[m + i];

    int u_it = nodes[u], u_end = nodes[u + 1];
    int v_it = nodes[v], v_end = nodes[v + 1];

    int a = edges[u_it], b = edges[v_it];
    while (u_it < u_end && v_it < v_end) {
      int d = a - b;
      if (d <= 0)
        a = edges[++u_it];
      if (d >= 0)
        b = edges[++v_it];
      if (d == 0)
        ++count;
    }
  }

  results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

void CudaAssert(cudaError_t status, const char* code, const char* file, int l) {
  if (status == cudaSuccess) return;
  cerr << "Cuda error: " << code << ", file " << file << ", line " << l << endl;
  exit(1);
}

#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

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



uint64_t MultiGPUCalculateTriangles(
    int n, int m, int* dev_edges, int* dev_nodes, int device_count) {
  vector<int*> multi_dev_edges(device_count);
  vector<int*> multi_dev_nodes(device_count);

  multi_dev_edges[0] = dev_edges;
  multi_dev_nodes[0] = dev_nodes;

  for (int i = 1; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaMalloc(&multi_dev_edges[i], m * 2 * sizeof(int)));
    CUCHECK(cudaMalloc(&multi_dev_nodes[i], (n + 1) * sizeof(int)));
    int dst = i, src = (i + 1) >> 2;
    CUCHECK(cudaMemcpyPeer(
          multi_dev_edges[dst], dst, multi_dev_edges[src], src,
          m * 2 * sizeof(int)));
    CUCHECK(cudaMemcpyPeer(
          multi_dev_nodes[dst], dst, multi_dev_nodes[src], src,
          (n + 1) * sizeof(int)));
  }

  vector<int> NUM_BLOCKS(device_count);
  vector<uint64_t*> multi_dev_results(device_count);

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    NUM_BLOCKS[i] = NUM_BLOCKS_PER_MP * NumberOfMPs();
    CUCHECK(cudaMalloc(
          &multi_dev_results[i],
          NUM_BLOCKS[i] * NUM_THREADS * sizeof(uint64_t)));
  }

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaFuncSetCacheConfig(CalculateTriangles, cudaFuncCachePreferL1));
    CalculateTriangles<<<NUM_BLOCKS[i], NUM_THREADS>>>(
        m, multi_dev_edges[i], multi_dev_nodes[i], multi_dev_results[i],
        device_count, i);
  }

  uint64_t result = 0;

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaDeviceSynchronize());
    result += SumResults(NUM_BLOCKS[i] * NUM_THREADS, multi_dev_results[i]);
  }

  for (int i = 1; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaFree(multi_dev_edges[i]));
    CUCHECK(cudaFree(multi_dev_nodes[i]));
  }

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaFree(multi_dev_results[i]));
  }

  cudaSetDevice(0);
  return result;
}

uint64_t GpuForward(int* edges, int num_nodes, uint64_t num_edges) {
  return MultiGpuForward(edges, 1, num_nodes, num_edges);
}

uint64_t GpuForward_v2(const TrCountingGraph& TrCountingGraph){
    int64_t* dev_offset;
    int* dev_neighbor;
    int* dev_length;
    int* dev_neighbor_start;
    CUCHECK(cudaMalloc(&dev_offset, (TrCountingGraph.nodeid_max + 2) * sizeof(int64_t)));
    CUCHECK(cudaMemcpyAsync(
       dev_offset, TrCountingGraph.offset, (TrCountingGraph.nodeid_max + 2) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUCHECK(cudaDeviceSynchronize());
    CUCHECK(cudaMalloc(&dev_length, (TrCountingGraph.nodeid_max + 1) * sizeof(int)));
    CUCHECK(cudaMemcpyAsync(
      dev_length, TrCountingGraph.degree, (TrCountingGraph.nodeid_max + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUCHECK(cudaDeviceSynchronize());
    CUCHECK(cudaMalloc(&dev_neighbor, ( TrCountingGraph.edge_num) * sizeof(int)));
    CUCHECK(cudaMemcpyAsync(
       dev_neighbor, TrCountingGraph.neighboor, ( TrCountingGraph.edge_num) * sizeof(int), cudaMemcpyHostToDevice));
    CUCHECK(cudaDeviceSynchronize());
    CUCHECK(cudaMalloc(&dev_neighbor_start, ( TrCountingGraph.edge_num) * sizeof(int)));
    CUCHECK(cudaMemcpyAsync(
       dev_neighbor_start, TrCountingGraph.neighboor_start, ( TrCountingGraph.edge_num) * sizeof(int), cudaMemcpyHostToDevice));
    CUCHECK(cudaDeviceSynchronize());
    const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();	
    uint64_t* dev_results;
    CUCHECK(cudaMalloc(&dev_results,
          NUM_BLOCKS * NUM_THREADS * sizeof(uint64_t)));
    
    cudaSetDevice(0);
    cudaFuncSetCacheConfig(CalculateTriangles_v2, cudaFuncCachePreferL1);
    CalculateTriangles_v2<<<NUM_BLOCKS, NUM_THREADS>>>(
        TrCountingGraph.offset[TrCountingGraph.nodeid_max+1], dev_neighbor, dev_offset, dev_length, dev_neighbor_start, dev_results);
    CUCHECK(cudaDeviceSynchronize());
    uint64_t result = SumResults(NUM_BLOCKS * NUM_THREADS, dev_results);
    return result;
}

int GetSplitNum(int num_nodes, uint64_t num_edges) {
  uint64_t mem = (uint64_t)GlobalMemory();  // in Byte
  mem -= (uint64_t)num_nodes * 16;  // uint64_t
  return (1 + 12 * (num_edges) / mem);
}

void InitializeGPUMemory() {
    
}

uint64_t GpuForwardSplit_v2(const TrCountingGraph& TrCountingGraph, int split_num, int64_t cpu_offset,
    int gpu_offset_start, int gpu_offset_end) {
    CUCHECK(cudaSetDevice(0));
    const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();
  
    // Calculate chunk size
    int64_t* split_offset;
    int64_t chunk_length_max = get_split_v2(TrCountingGraph.offset, TrCountingGraph.nodeid_max, split_num, split_offset);

    // device memory initialization
    int64_t* dev_offset;
    int* dev_length;
    int* dev_neighbor_i;
    int* dev_neighbor_start_i;
    int* dev_neighbor_j;
    CUCHECK(cudaMalloc(&dev_offset, (TrCountingGraph.nodeid_max + 2) * sizeof(int64_t)));
    CUCHECK(cudaMemcpyAsync(
      dev_offset, TrCountingGraph.offset, (TrCountingGraph.nodeid_max + 2) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUCHECK(cudaDeviceSynchronize());
  CUCHECK(cudaMalloc(&dev_length, (TrCountingGraph.nodeid_max + 1) * sizeof(int)));
  CUCHECK(cudaMemcpyAsync(
    dev_length, TrCountingGraph.degree, (TrCountingGraph.nodeid_max + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUCHECK(cudaDeviceSynchronize());
  CUCHECK(cudaMalloc(&dev_neighbor_i, chunk_length_max * sizeof(int)));
  CUCHECK(cudaMalloc(&dev_neighbor_start_i, chunk_length_max * sizeof(int)));
  CUCHECK(cudaMalloc(&dev_neighbor_j, chunk_length_max * sizeof(int)));
  uint64_t* dev_results;
  CUCHECK(cudaMalloc(&dev_results, NUM_BLOCKS * NUM_THREADS * sizeof(uint64_t)));
  int64_t* dev_split_offset;
  CUCHECK(cudaMalloc(&dev_split_offset, (split_num + 1) * sizeof(int64_t)));
  CUCHECK(cudaMemcpyAsync(
    dev_split_offset, split_offset, (split_num + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUCHECK(cudaDeviceSynchronize());

  cudaSetDevice(0);
  cudaFuncSetCacheConfig(CalculateTrianglesSplit_v2, cudaFuncCachePreferL1);
  int64_t result=0;
  for(int ij = gpu_offset_start; ij < gpu_offset_end; ij++) {
      int i, j;
      get_i_j(split_num, ij, &i, &j);
      if(split_offset[i] >= cpu_offset)
          break;
      CUCHECK(cudaMemcpyAsync(
         dev_neighbor_i, TrCountingGraph.neighboor+split_offset[i], (split_offset[i+1]-split_offset[i])*sizeof(int), cudaMemcpyHostToDevice));
      CUCHECK(cudaMemcpyAsync(
        dev_neighbor_start_i, TrCountingGraph.neighboor_start+split_offset[i], (split_offset[i+1]-split_offset[i])*sizeof(int), cudaMemcpyHostToDevice));
      CUCHECK(cudaMemcpyAsync(
        dev_neighbor_j, TrCountingGraph.neighboor+split_offset[j], (split_offset[j+1]-split_offset[j])*sizeof(int), cudaMemcpyHostToDevice));
      CUCHECK(cudaDeviceSynchronize());

      CalculateTrianglesSplit_v2<<<NUM_BLOCKS, NUM_THREADS>>>(
        cpu_offset>split_offset[i+1]?split_offset[i+1]-split_offset[i]:cpu_offset-split_offset[i],
        dev_offset, dev_length, dev_neighbor_i, dev_neighbor_start_i, dev_neighbor_j, dev_split_offset, i, j, dev_results);
      CUCHECK(cudaDeviceSynchronize());
      result = result + SumResults(NUM_BLOCKS * NUM_THREADS, dev_results);
  }
  return result;
}

uint64_t MultiGpuForward(int* edges, int device_count, int num_nodes, uint64_t num_edges) {
  CUCHECK(cudaSetDevice(0));
  const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();

  uint64_t m = num_edges;
  int n = num_nodes;

  int* dev_edges;
  int* dev_nodes;

  
  int* dev_temp;
  CUCHECK(cudaMalloc(&dev_temp, m * 2 * sizeof(int)));
  CUCHECK(cudaMemcpyAsync(
      dev_temp, edges, m * 2 * sizeof(int), cudaMemcpyHostToDevice));
  CUCHECK(cudaDeviceSynchronize());
  // Memcpy edges from host to device
  SortEdges(m, dev_temp);
  CUCHECK(cudaDeviceSynchronize());
  // Sort edges

  CUCHECK(cudaMalloc(&dev_edges, m * 2 * sizeof(int)));
  UnzipEdges<<<NUM_BLOCKS, NUM_THREADS>>>(m, dev_temp, dev_edges);
  CUCHECK(cudaFree(dev_temp));
  CUCHECK(cudaDeviceSynchronize());
  // Unzip edges


  CUCHECK(cudaMalloc(&dev_nodes, (n + 1) * sizeof(int)));
  CalculateNodePointers<false><<<NUM_BLOCKS, NUM_THREADS>>>(
      n, m, dev_edges, dev_nodes);
  CUCHECK(cudaDeviceSynchronize());
  // Calculate nodes array for one-way unzipped edges
  uint64_t result = 0;

  if (device_count == 1) {
    uint64_t* dev_results;
    CUCHECK(cudaMalloc(&dev_results,
          NUM_BLOCKS * NUM_THREADS * sizeof(uint64_t)));
    cudaFuncSetCacheConfig(CalculateTriangles, cudaFuncCachePreferL1);
    cudaProfilerStart();
    CalculateTriangles<<<NUM_BLOCKS, NUM_THREADS>>>(
        m, dev_edges, dev_nodes, dev_results);
    CUCHECK(cudaDeviceSynchronize());
    cudaProfilerStop();
    // Reduce
    result = SumResults(NUM_BLOCKS * NUM_THREADS, dev_results);
    CUCHECK(cudaFree(dev_results));
  } else {
    result = MultiGPUCalculateTriangles(
        n, m, dev_edges, dev_nodes, device_count);
  }

  CUCHECK(cudaFree(dev_edges));
  CUCHECK(cudaFree(dev_nodes));
  return result;
}

void PreInitGpuContext(int device) {
  CUCHECK(cudaSetDevice(device));
  CUCHECK(cudaFree(NULL));
}
