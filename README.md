Triangles Counting in CUDA GPU
=========
[![Build Status](https://travis-ci.com/zhaofeng-shu33/triangle_counting_gpu_v2.svg?branch=master)](https://travis-ci.com/zhaofeng-shu33/triangle_counting_gpu_v2)

CUDA implementation of parallel algorithm for counting triangles.

## History
This repository is spawned out of [triangle counting gpu](https://github.com/zhaofeng-shu33/triangle_counting_gpu).
[Zhiyuan-Wu](https://github.com/Zhiyuan-Wu) implements it in a different approach and cannot be merged into master branch. Therefore this repository is created from the `c-factor` branch. Generally speaking, this repository is faster than the original implementation but it also brings some drawbacks:

* CPU GPU coordinate parameter is hard coded
* Thread Number, Batch size is hard coded
* Using unix only header and is difficult to port to msvc on windows

## CSR format storage
Suppose our graph has m edges and n nodes.
we use sparse matrix (CSC) to store the graph. Two arrays are required, (n+1) length array
(row array) and (m) length array storing node index. We only store upper triangular matrix.

For example, suppose our graph has (0,1),(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(4,5).
n=6,m=8.
Storing in CSR Lower Triangular Format is equivalent to:
`[0,1,1,2,2,3,3,4]`(m) and `[0,0,1,2,4,6,8]`(n+1).

```Python
from scipy import sparse
a = sparse.csr_matrix(([1]*8,[0,1,1,2,2,3,3,4], [0,0,1,2,4,6,8]), shape=(6,6))
print(a)
```

## Counting Triangles
The parallelism comes from the triangles containing of each edge is independent from each other.
 
