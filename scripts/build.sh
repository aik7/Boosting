#!/bin/bash

BOOSTING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${BOOSTING_DIR}

export BOOSTING_EXT_DIR=${BOOSTING_DIR}"/external"


# process the command line argument
while getopts b:g: flag
do
  case "${flag}" in
    b) build_type=${OPTARG};;
    g) gurobi_option=${OPTARG};;
  esac
done


# set CMAKE_BUILD_TYPE
if [ "${build_type}" = "debug" ]; then
  export DEBUG_OPTION="-DCMAKE_BUILD_TYPE=Debug"
else
  export DEBUG_OPTION="-DCMAKE_BUILD_TYPE=Release"
fi

echo ${DEBUG_OPTION}


# set Gurobi
if [ "${gurobi_option}" = "gurobi" ]; then
  export GUROBI_OPTIONS="-DENABLE_GUROBI=true"
else
  export GUROBI_OPTIONS="-DENABLE_GUROBI=false"
fi

echo ${GUROBI_OPTIONS}


# build PEBBL and RMA
cd ${BOOSTING_DIR}"/external/RMA/"
sh scripts/build.sh
# Ai: decided not to pass the build type to RMA build since it takes time to rebuild
# sh scripts/build.sh -b ${build_type}


# Build Coin-OR CLP if it does not exits
if [ -d ${BOOSTING_EXT_DIR}"/coin" ]; then
  echo "DIRECTORY ${BOOSTING_EXT_DIR}/coin EXITS"
else
  cd ${BOOSTING_EXT_DIR}
  # Build Coin-OR CLP
  sh build_clp.sh
fi


# if GUROBI_HOME directory exists
if [ -d "${GUROBI_HOME}" ]; then
  echo "Using GUROBI_HOME specified in bahsrc"
# else if gurobi option is enabled and gurobi folder is empty, download gurobi
elif [ "${gurobi_option}" = "gurobi" ]  && [ -d ${BOOSTING_EXT_DIR}"/gurobi" ]; then
    cd ${BOOSTING_EXT_DIR}
    sh build_gurobi.sh
else
  echo "DIRECTORY ${BOOSTING_EXT_DIR}/gurobi EXITS"
fi


# build Boosting
export BOOSTING_BUILD_DIR=${BOOSTING_DIR}"/build"
mkdir -p ${BOOSTING_BUILD_DIR}
cd ${BOOSTING_BUILD_DIR}
cmake ${DEBUG_OPTION} ${GUROBI_OPTIONS} ..
make
