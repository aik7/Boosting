#!/bin/bash

# go to the RMA directory
cd external/RMA

# build PEBBL and RMA
sh scripts/build.sh

# go to the external direcoty
cd ../

# Build Coin-OR
sh build_clp.sh

# Build Gurobi
sh build_gurobi.sh

# build Boosting
cd ../
mkdir build
cd build
cmake -DENABLE_GUROBI=true ..
make
