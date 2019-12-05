#!/bin/bash
set -e -x
mkdir build
cd build
cmake .. # GPU compilation is on
make
