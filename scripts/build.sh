#!/bin/bash

# build PEBBL and RMA

cd external/RMA
sh scripts/build.sh

# Build Coin-OR

cd ../
sh build_clp.sh

# build Boosting
cd ../
mkdir build
cd build
cmake ..
make
