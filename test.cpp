// Copyright 2019 zhaofeng-shu33

#include <iostream>

#include <gtest/gtest.h>  // NOLINT(build/include_order)

#include <nvtc/graph.h>
#include <nvtc/gpu.h>
#include <nvtc/counting_cpu.h>

#if GPU
TEST(tc, io_bin) {
    int* edges;
    std::pair<int, uint64_t> info_pair = read_binfile_to_arclist_v2("test_io.bin", edges);
    uint64_t trcount = GpuForward(edges, info_pair.first, info_pair.second);
    EXPECT_EQ(trcount, 1); 
    free(edges);
    EXPECT_THROW(read_binfile_to_arclist_v2("test_io_false.bin", edges),
                 std::logic_error);
}

TEST(tc, io_nvgraph) {
    int* edges;
    std::pair<int, uint64_t> info_pair = read_binfile_to_arclist_v2("test_io_nvgraph.bin", edges);
    uint64_t trcount = GpuForward(edges, info_pair.first, info_pair.second);
    EXPECT_EQ(trcount, 3); 
    free(edges);
}

TEST(tcv2, io_bin) {
    MyGraph myGraph("test_io.bin");
    uint64_t tcount = GpuForward_v2(myGraph);
    EXPECT_EQ(tcount, 1);
}

TEST(tcv2, io_nvgraph) {
    MyGraph myGraph("test_io_nvgraph.bin");
    uint64_t tcount = GpuForward_v2(myGraph);
    EXPECT_EQ(tcount, 3);
}
#endif

TEST(cpu, io_bin) {
    int* arcs;
    std::pair<int, uint64_t> info_pair = read_binfile_to_arclist_v2("test_io.bin", arcs);
    uint64_t tcount = CpuForward(arcs, info_pair.first, info_pair.second);
    EXPECT_EQ(tcount, 1);    
    free(arcs);
}

TEST(cpu, io_nvgraph) {
    int* arcs;
    std::pair<int, uint64_t> info_pair = read_binfile_to_arclist_v2("test_io_nvgraph.bin", arcs);
    uint64_t tcount = CpuForward(arcs, info_pair.first, info_pair.second);
    EXPECT_EQ(tcount, 3);    
    free(arcs);
}


