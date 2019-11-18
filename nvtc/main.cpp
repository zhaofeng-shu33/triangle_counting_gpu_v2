#include "gpu.h"
#include "graph.h"
#include "timer.h"
#include "counting_cpu.h"
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>

using namespace std;


int main(int argc, char *argv[]) {
    if (argc != 3 || strcmp(argv[1], "-f") != 0) {
        std::cout << "Usage: nvtc-variant -f input.bin" << std::endl;
        exit(-1);
    }
#if TIMECOUNTING 
    unique_ptr<Timer> t(Timer::NewTimer());
#endif
#if SECONDVERSION
    MyGraph myGraph(argv[2]);
#else    
    const char* io_hint = std::getenv("DATAIO");
    const char* device_hint = std::getenv("DEVICEHINT");
    int* edges;
    std::pair<int, uint64_t> info_pair;
    info_pair = read_binfile_to_arclist_v2(argv[2], edges);
#if VERBOSE    
    std::cout << "Num of Nodes: " << info_pair.first << std::endl;
    std::cout << "Num of Edges: " << info_pair.second << std::endl;
#endif
#endif

#if TRCOUNTING
    uint64_t result = 0;
    t->Done("Reading Data");
#if SECONDVERSION
    int64_t cpu_split_target = (int64_t) ((double)(myGraph.offset[myGraph.nodeid_max+1]) * 0.01);
    int64_t cpu_offset = *lower_bound(myGraph.offset,myGraph.offset+myGraph.nodeid_max+2,cpu_split_target);
    int split_num = GetSplitNum(myGraph.nodeid_max,myGraph.offset[myGraph.nodeid_max+1],cpu_offset);
    cout<<"GPU Split: "<<split_num<<" Submit..."<<endl;
    if (split_num>1){
        int64_t cpu_result = 0;
        cout<<"CPU Task: "<<cpu_offset<<"/"<<myGraph.offset[myGraph.nodeid_max+1]<<" Start..."<<endl;
        thread cpu_thread(cpu_counting_edge_first_v2,&myGraph,cpu_offset,&cpu_result);
        result = GpuForwardSplit_v2(myGraph,split_num,cpu_offset);
        cout<<"GPU Done."<<endl;
        cpu_thread.join();
        result = result + cpu_result;
    }
    else{
        result = GpuForward_v2(myGraph);
    }
#else
#if GPU
    if(device_hint == NULL || strcmp(device_hint, "GPU") == 0){
       result = GpuForward(edges, info_pair.first, info_pair.second);
    }
    else if (strcmp(device_hint, "CPU") == 0) {
        result = CpuForward(edges, info_pair.first, info_pair.second);
    } else {
        result = GpuForward(edges, info_pair.first, info_pair.second);
    }
#else
	result = CpuForward(edges, info_pair.first, info_pair.second);
#endif
    free(edges);
#endif
#if TIMECOUNTING    
    t->Done("Compute number of triangles");
#endif
    cout << "There are " << result <<
            " triangles in the input graph." << endl;
#endif
}
