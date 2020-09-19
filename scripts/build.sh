#!/bin/bash

# build PEBBL and RMA

cd external/RMA
sh scripts/build.sh

# Build Coin-OR

cd ../
sh build.sh

# build Boosting
cd ../
mkdir build
cd build
cmake ..
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
