#ifndef GPU_H
#define GPU_H

#include "graph.h"
#include "MyGraph.h"
#include <stdint.h>

uint64_t GpuForward(const Edges& edges);
uint64_t GpuForward_v2(const MyGraph& myGraph);
uint64_t MultiGpuForward(const Edges& edges, int device_count);
uint64_t CpuForward(const Edges& edges, int node_num);
void PreInitGpuContext(int device = 0);

#endif
