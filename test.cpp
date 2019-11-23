// Copyright 2019 zhaofeng-shu33
#include <stdio.h>
#include <gtest/gtest.h>  // NOLINT(build/include_order)

#include <nvtc/TrCountingGraph.h>
#include <nvtc/gpu.h>

#if GPU
TEST(tcv2, io_bin) {
    TrCountingGraph TrCountingGraph("test_io.bin");
    uint64_t tcount = GpuForward_v2(TrCountingGraph);
    EXPECT_EQ(tcount, 1);
}

TEST(tcv2, io_nvgraph) {
    TrCountingGraph TrCountingGraph("test_io_nvgraph.bin");
    uint64_t tcount = GpuForward_v2(TrCountingGraph);
    EXPECT_EQ(tcount, 3);
}
#endif


TEST(io, get_edge_num) {
   FILE* pFile = fopen("test_io.bin", "rb");
   int64_t size = get_edge_num(pFile);
   fclose(pFile);
   EXPECT_EQ(size, 3);
}

