#include "TrCountingGraph.h"
#if GPU
#include "gpu.h"
#endif
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <memory>
#include <vector>
#include <thread>
#include <inttypes.h>
#include <unistd.h>
#if USEMPI
#include "mpi.h"
#endif
using namespace std;



int main(int argc, char *argv[]) {
#if USEMPI
    int numtasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
#endif
        if (argc != 3 || strcmp(argv[1], "-f") != 0) {
            printf("Usage: nvtc-variant -f input.bin\n");
            exit(-1);
        }
#if USEMPI
    }
#endif
    char* file_name = argv[2];
    if (access(file_name, F_OK ) == -1) {
#if USEMPI
    if (rank == 0) {
#endif
        printf("file %s file_name does not exist\n", file_name);
#if USEMPI
    }
#endif
        exit(-1);
    }
    TrCountingGraph trCountingGraph(file_name);

#if TRCOUNTING
    int64_t result = 0;
    int64_t cpu_offset_end = trCountingGraph.offset[trCountingGraph.nodeid_max + 1];
#if GPU    
    // Compute Split Information
    int split_num = GetSplitNum(trCountingGraph.nodeid_max,trCountingGraph.offset[trCountingGraph.nodeid_max+1]);
    int64_t* split_offset;
    int64_t chunk_length_max = get_split_v2(trCountingGraph.offset, trCountingGraph.nodeid_max, split_num, split_offset);
    
    // Last k% edges will be calculated by cpu.
    int64_t cpu_offset_start = (int64_t) ((double)(cpu_offset_end) * (1-0.02));
    if (split_num > 1) {
        int64_t cpu_result = 0;
        thread cpu_thread(cpu_counting_edge_first_v2, &trCountingGraph, cpu_offset_start, cpu_offset_end, &cpu_result);
        result = GpuForwardSplit_v2(trCountingGraph, split_num, cpu_offset_start);        
        cpu_thread.join();
        result = result + cpu_result;
    }
    else {
        result = GpuForward_v2(trCountingGraph);
    }
#else
#if USEMPI
    int64_t cpu_offset_rank_start = rank * cpu_offset_end / numtasks;
    int64_t cpu_offset_rank_end = (rank + 1) * cpu_offset_end / numtasks;
    cpu_counting_edge_first_v2(&trCountingGraph, cpu_offset_rank_start, cpu_offset_rank_end, &result);
    if (rank == 0) {
        // receive computing results from rank > 1
        int64_t result_other_node;
        MPI_Status Stat;
        for(int i = 1; i < numtasks; i++) {
            MPI_Recv(&result_other_node, 1, MPI_INT64_T, i, 1, MPI_COMM_WORLD, &Stat);
            result += result_other_node;
        }
    }
    else {
       // send computing results to node with rank = 0
       MPI_Send(&result, 1, MPI_INT64_T, 0, 1, MPI_COMM_WORLD);
    }
#else
    cpu_counting_edge_first_v2(&trCountingGraph, 0, cpu_offset_end, &result);
#endif
#endif
#if USEMPI
    if (rank == 0) {
#endif
        printf("There are %" PRId64 " triangles in the input graph.\n", result);
#if USEMPI
    }
#endif    
#endif

#if USEMPI
    MPI_Finalize();
#endif
}
