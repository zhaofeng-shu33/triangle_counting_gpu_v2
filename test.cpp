// Copyright 2019 zhaofeng-shu33

#include <iostream>

#include <gtest/gtest.h>  // NOLINT(build/include_order)

#include <nvtc/graph.h>
#include <nvtc/gpu.h>



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
