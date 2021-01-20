#!/bin/bash


# Build Gurobi
cd external
sh build_gurobi.sh

# build Boosting

cd ../
mkdir build
cd build
cmake -DENABLE_GUROBI=true ..
make
