#ifndef GPU_H
#define GPU_H

#include "graph.h"
#include "MyGraph.h"
#include <stdint.h>

int GetSplitNum(int num_nodes, uint64_t num_edges);
uint64_t GpuForward_v2(const MyGraph& myGraph);
uint64_t GpuForwardSplit_v2(const MyGraph& myGraph, int split_num);
void PreInitGpuContext(int device = 0);
uint64_t GpuForward(int* edges, int num_nodes, uint64_t num_edges);
uint64_t MultiGpuForward(int* edges, int device_count, int num_nodes, uint64_t num_edges);
#endif
