#include "gpu.h"
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
    MyGraph myGraph(argv[2]);

#if TRCOUNTING
    uint64_t result = 0;
#if TIMECOUNTING
    t->Done("Reading Data");
#endif
    // Compute Split Information
    int split_num = GetSplitNum(myGraph.nodeid_max,myGraph.offset[myGraph.nodeid_max+1]);
    int64_t* split_offset;
    int64_t chunk_length_max = get_split_v2(myGraph.offset, myGraph.nodeid_max, split_num, split_offset);
    
    // Last k% edges will be calculated by cpu.
    int64_t cpu_offset = (int64_t) ((double)(myGraph.offset[myGraph.nodeid_max+1]) * (1-0.05));
    if (split_num>1){
        int64_t cpu_result = 0;
        thread cpu_thread(cpu_counting_edge_first_v2,&myGraph,cpu_offset,&cpu_result);
        result = GpuForwardSplit_v2(myGraph,split_num,cpu_offset);
        cout<<"GPU Done."<<endl;
        cpu_thread.join();
        result = result + cpu_result;
    }
    else{
        result = GpuForward_v2(myGraph);
    }
#if TIMECOUNTING    
    t->Done("Compute number of triangles");
#endif
    cout << "There are " << result <<
            " triangles in the input graph." << endl;
#endif
}
