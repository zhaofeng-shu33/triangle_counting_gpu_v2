#ifndef GPU_H
#define GPU_H

#include "graph.h"
#include "TrCountingGraph.h"
#include <stdint.h>

int GetSplitNum(int num_nodes, uint64_t num_edges);
uint64_t GpuForward_v2(const TrCountingGraph& TrCountingGraph);
uint64_t GpuForwardSplit_v2(const TrCountingGraph& TrCountingGraph, int split_num, int64_t cpu_offset);
void PreInitGpuContext(int device = 0);
uint64_t GpuForward(int* edges, int num_nodes, uint64_t num_edges);
uint64_t MultiGpuForward(int* edges, int device_count, int num_nodes, uint64_t num_edges);
#endif
