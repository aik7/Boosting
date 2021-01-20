#!/bin/bash

# build PEBBL and RMA

cd external/RMA
sh scripts/build_debug.sh

# Build Coin-OR

cd ../
sh build_clp.sh

# build Boosting
cd ../
mkdir build
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
make


# #  Build PEBBL

# mkdir external/RMA/external/pebbl/build
# cd    external/RMA/external/pebbl/build
# cmake -Denable_mpi=ON -Denable_examples=OFF ..
# make


# # Build RMA

# cd ../../../  # go back to the RMA root directory

# mkdir build
# cd build
# cmake ..
# make
