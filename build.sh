#!/bin/bash
bash /etc/profile.d/modules.sh
module load cuda10.0/toolkit slurm
set -e -x
mkdir build
python3 graph_generator.py
cd build
cmake3 -DENABLE_TESTING=ON ..
make -j5
srun -t 500 --gres=gpu:1 test_alg

## test other configuration 
cmake3 -DENABLE_TESTING=OFF -DVERBOSE=ON ..
make -j5

cmake3 -DMPI=ON ..
make -j5

cmake3 -DOPENMP=ON ..
make -j5

cmake3 -DTRCOUNTING=OFF ..
make -j5

cmake3 -DTRCOUNTING=ON -DCODEGPU=OFF ..
make -j5
