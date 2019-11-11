// Copyright 2019 zhaofeng-shu33

#include <iostream>

#include <gtest/gtest.h>  // NOLINT(build/include_order)

#include <nvtc/graph.h>
#include <nvtc/gpu.h>
#include <nvtc/counting_cpu.h>

#if GPU
TEST(tc, io_bin) {
    Edges edges;
    ReadEdgesFromFile("test_io.bin", edges);
    uint64_t trcount = GpuForward(edges);
    EXPECT_EQ(trcount, 1); 
    edges.clear();
    EXPECT_THROW(ReadEdgesFromFile("test_io_false.bin", edges),
                 std::logic_error);
}

TEST(tc, io_nvgraph) {
    Edges edges;
    ReadEdgesFromFile("test_io_nvgraph.bin", edges);
    uint64_t trcount = GpuForward(edges);
    EXPECT_EQ(trcount, 3); 
}
#if SECONDVERSION
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
#endif

TEST(cpu, io_bin) {
    std::vector<std::pair<int, int>> arcs;
    std::pair<int, int> info_pair = read_binfile_to_arclist("test_io.bin", arcs);
    uint64_t tcount = CpuForward(arcs, info_pair.first);
    EXPECT_EQ(tcount, 1);    
}

TEST(cpu, io_nvgraph) {
    std::vector<std::pair<int, int>> arcs;
    std::pair<int, int> info_pair = read_binfile_to_arclist("test_io_nvgraph.bin", arcs);
    uint64_t tcount = CpuForward(arcs, info_pair.first);
    EXPECT_EQ(tcount, 3);    
}

TEST(dataio, io_bin) {
   std::vector<std::pair<int, int>> arcs;
   std::pair<int, int> info_pair = read_binfile_to_arclist("test_io.bin", arcs);
   std::vector<std::pair<int, int>> arcs_v2;
   std::pair<int, int> info_pair_v2 = read_binfile_to_arclist_v2("test_io.bin", arcs_v2);
   EXPECT_EQ(info_pair, info_pair_v2);
   EXPECT_EQ(arcs, arcs_v2);
}

TEST(dataio, io_nvgraph) {
   std::vector<std::pair<int, int>> arcs;
   std::pair<int, int> info_pair = read_binfile_to_arclist("test_io_nvgraph.bin", arcs);
   std::vector<std::pair<int, int>> arcs_v2;
   std::pair<int, int> info_pair_v2 = read_binfile_to_arclist_v2("test_io_nvgraph.bin", arcs_v2);
   EXPECT_EQ(info_pair, info_pair_v2);
}

