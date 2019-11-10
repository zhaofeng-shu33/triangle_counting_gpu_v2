#include "cpu.h"
#include "gpu.h"
#include "graph.h"
#include "timer.h"

#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;


int main(int argc, char *argv[]) {
    if (argc != 3 || strcmp(argv[1], "-f") != 0) {
        std::cout << "Usage: lemon-tc -f input.bin" << std::endl;
        exit(-1);
    }
#if TIMECOUNTING 
    unique_ptr<Timer> t(Timer::NewTimer());
#endif
    Edges edges = ReadEdgesFromFile(argv[2]);
#if TIMECOUNTING   
    t->Done("Read file");
    t->Done("Convert to adjacency lists");
#endif


#if TIMECOUNTING
    t->Reset();
    PreInitGpuContext(0);
    t->Done("Preinitialize context for device 0");
#endif
    uint64_t result = GpuForward(edges);

    cout << "There are " << result <<
            " triangles in the input graph." << endl;
}
