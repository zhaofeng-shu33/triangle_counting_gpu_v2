// Copyright 2019 zhaofeng-shu33
#include <stdio.h>
#include <gtest/gtest.h>  // NOLINT(build/include_order)
#include <nvtc/config.h>
#include <nvtc/TrCountingGraph.h>
#if GPU
#include <nvtc/gpu.h>
#endif


TEST(tcv2, io_bin) {
    TrCountingGraph trCountingGraph("test_io.bin");
    int64_t tcount;
    int64_t offset_end = trCountingGraph.offset[trCountingGraph.nodeid_max + 1];
    cpu_counting_edge_first_v2(&trCountingGraph, 0, offset_end, &tcount);
    EXPECT_EQ(tcount, 1);
#if GPU
    tcount = GpuForward_v2(trCountingGraph);
    EXPECT_EQ(tcount, 1);
#endif
}

TEST(tcv2, io_nvgraph) {
    TrCountingGraph trCountingGraph("test_io_nvgraph.bin");
#if GPU
    uint64_t tcount = GpuForward_v2(trCountingGraph);
    EXPECT_EQ(tcount, 3);
    tcount = GpuForwardSplit_v2(trCountingGraph, 2, trCountingGraph.offset[trCountingGraph.nodeid_max + 1], 0, 4);
#endif
}


TEST(io, get_edge_num) {
   FILE* pFile = fopen("test_io.bin", "rb");
   int64_t size = get_edge_num(pFile);
   fclose(pFile);
   EXPECT_EQ(size, 3);
}

TEST(io, get_max_id) {
   FILE* pFile = fopen("test_io.bin", "rb");
   int* data = (int*)malloc(sizeof(int) * 6);
   fread(data, sizeof(int), 6, pFile);
   int max_id = get_max_id(data, 6);
   free(data);
   fclose(pFile);
   EXPECT_EQ(max_id, 2);
}

TEST(utility, ijpair) {
   int i, j;
   get_i_j(6, 35, &i, &j);
   EXPECT_EQ(i, 5);
   EXPECT_EQ(j, 5);
   get_i_j(6, 10, &i, &j);
   EXPECT_EQ(i, 1);
   EXPECT_EQ(j, 4);
}
