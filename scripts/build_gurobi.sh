#!/bin/bash


# Build Gurobi
cd external
sh build_gurobi.sh

# build Boosting
cd ../
cd build
rm -rf *
cmake -DENABLE_GUROBI=true ..
make
