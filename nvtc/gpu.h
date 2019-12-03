#ifndef GPU_H
#define GPU_H

#include "TrCountingGraph.h"
#include <stdint.h>

int GetSplitNum(int num_nodes, uint64_t num_edges);
uint64_t GpuForward_v2(const TrCountingGraph& TrCountingGraph);
uint64_t GpuForwardSplit_v2(const TrCountingGraph& TrCountingGraph, int split_num, int64_t cpu_offset, int gpu_offset_start, int gpu_offset_end);
void PreInitGpuContext(int device = 0);
uint64_t GpuForward(int* edges, int num_nodes, uint64_t num_edges);
uint64_t MultiGpuForward(int* edges, int device_count, int num_nodes, uint64_t num_edges);

class TrCountingGraphChunk{
    public:
        TrCountingGraphChunk(const TrCountingGraph &g, int split_num);
        void initChunk(int i, int j);
        void initAll();
        const TrCountingGraph* Graph;
        int64_t* split_offset;
        int split_num;
        int64_t chunk_length_max;

        int64_t* dev_offset;
        int* dev_degree;
        int* dev_neighbor_i;
        int* dev_neighbor_start_i;
        int* dev_neighbor_j;
        int64_t* dev_split_offset;
        int* dev_neighbor;
        int* dev_neighbor_start;
        int chunkid_i;
        int chunkid_j;
};
#endif
