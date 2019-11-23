#!/bin/bash
set -e -x
SYSTEMNAME=$(cat /etc/issue)
FLAGS=`echo $SYSTEMNAME | awk '{print match($0,"Ubuntu")}'`
if [ $FLAGS -gt 0 ]; then
    export PATH=/usr/local/lib:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}
    mkdir build
    python graph_generator.py
    cd build
    cmake -DENABLE_TESTING=ON ..
    make -j5
    ./test_alg
else
    bash /etc/profile.d/modules.sh
    module load cuda10.0/toolkit slurm
    mkdir build
    python3 graph_generator.py
    cd build
    cmake3 -DENABLE_TESTING=ON ..
    make -j5
    srun -t 500 --gres=gpu:1 test_alg
fi
