#include "gpu.h"
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <memory>
#include <vector>
#include <thread>
#include <inttypes.h>

using namespace std;


int main(int argc, char *argv[]) {
    if (argc != 3 || strcmp(argv[1], "-f") != 0) {
        printf("Usage: nvtc-variant -f input.bin\n");
        exit(-1);
    }
    TrCountingGraph TrCountingGraph(argv[2]);

#if TRCOUNTING
    uint64_t result = 0;
    // Compute Split Information
    int split_num = GetSplitNum(TrCountingGraph.nodeid_max,TrCountingGraph.offset[TrCountingGraph.nodeid_max+1]);
    int64_t* split_offset;
    int64_t chunk_length_max = get_split_v2(TrCountingGraph.offset, TrCountingGraph.nodeid_max, split_num, split_offset);
    
    // Last k% edges will be calculated by cpu.
    int64_t cpu_offset = (int64_t) ((double)(TrCountingGraph.offset[TrCountingGraph.nodeid_max+1]) * (1-0.05));
    if (split_num>1){
        int64_t cpu_result = 0;
        thread cpu_thread(cpu_counting_edge_first_v2,&TrCountingGraph,cpu_offset,&cpu_result);
        result = GpuForwardSplit_v2(TrCountingGraph,split_num,cpu_offset);        
        cpu_thread.join();
        result = result + cpu_result;
    }
    else{
        result = GpuForward_v2(TrCountingGraph);
    }
    printf("There are " PRIu64 " triangles in the input graph.\n");
#endif
}
