cmake_minimum_required(VERSION 3.0)
project(boosting)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#include(ExternalProject)
#set( EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external_install )
#add_subdirectory(external)

set(PEBBL_ROOT ${CMAKE_SOURCE_DIR}/external/RMA/external/pebbl)
set(RMA_ROOT   ${CMAKE_SOURCE_DIR}/external/RMA)
set(Coin_ROOT  ${CMAKE_SOURCE_DIR}/external/coin)

set(PEBBL_LIBRARIES ${PEBBL_ROOT}/build/src/pebbl/libpebbl.a)
set(PEBBL_INCLUDES  ${PEBBL_ROOT}/src ${PEBBL_ROOT}/build/src)

set(RMA_LIBRARIES ${RMA_ROOT}/build/liblrma.a)
set(RMA_INCLUDES  ${RMA_ROOT}/src)

set(Coin_INCLUDES ${Coin_ROOT}/Clp/src
                  ${Coin_ROOT}/CoinUtils/src
                  ${Coin_ROOT}/build/Clp/master/src/
                  ${Coin_ROOT}/build/CoinUtils/master/src/
                   )
set(Coin_LIBRARIES ${Coin_ROOT}/dist/lib/libClp.so
                   ${Coin_ROOT}/dist/lib/libCoinUtils.so
                   )

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
  # set(CMAKE_BUILD_TYPE "Debug")
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall -Werror=reorder")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

MESSAGE( STATUS "CMAKE_BUILD_TYPE        : ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "C++ compiler            : ${CMAKE_CXX_COMPILER}" )
MESSAGE( STATUS "C++ compiler ID         : ${CMAKE_CXX_COMPILER_ID}" )
MESSAGE( STATUS "C++ compiler version    : ${CMAKE_CXX_COMPILER_VERSION}" )
MESSAGE( STATUS "C++ compiler string     : ${CXX_COMPILER_STRING}" )
MESSAGE( STATUS "MPI C++ compiler        : ${MPI_CXX_COMPILER}" )
MESSAGE( STATUS "MPI C++ compiler options: ${MPI_CXX_COMPILE_OPTIONS}" )
MESSAGE( STATUS "MPI libraries           : ${MPI_LIBRARIES}")
MESSAGE( STATUS "PEBBL include dir       : ${PEBBL_INCLUDES}" )
MESSAGE( STATUS "PEBBL libraries         : ${PEBBL_LIBRARIES}")
MESSAGE( STATUS "CLP include dir         : ${Coin_INCLUDES}" )
MESSAGE( STATUS "CLP libraries           : ${Coin_LIBRARIES}")

option(ENABLE_GUROBI "Enable Gurobi build. Will require Gurobi wrappers
                      for compilers." OFF)

if(ENABLE_GUROBI)

  MESSAGE(  STATUS "GUROBI enabled.")
  set(HAVE_GUROBI 1)

  # Option 2: download Gurobi in external directory
  if(GUROBI_HOME STREQUAL "")
    set(GUROBI_ROOT ${CMAKE_SOURCE_DIR}/external/gurobi/gurobi911)

  # Option 1: already set GURROBI_HOME and GUROBI_LIBRARIES in the .bashrc files
  else()
    set(GUROBI_ROOT  $ENV{GUROBI_HOME}/..)
  endif()

  set(GUROBI_INCLUDES ${GUROBI_ROOT}/linux64/include
                      ${GUROBI_ROOT}/linux64/src)
  set(GUROBI_LIBRARIES ${GUROBI_ROOT}/linux64/lib/libgurobi_g++5.2.a
                       ${GUROBI_ROOT}/linux64/lib/libgurobi91.so)
  MESSAGE( STATUS "GUROBI include dir      : ${GUROBI_INCLUDES}" )
  MESSAGE( STATUS "GUROBI libraries        : ${GUROBI_LIBRARIES}")

endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/boosting_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/boosting_config.h)

include_directories(
  src
  ${MPI_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}
  ${PEBBL_INCLUDES}
  ${RMA_INCLUDES}
  ${Coin_INCLUDES}
  ${CMAKE_CURRENT_BINARY_DIR}
  if(ENABLE_GUROBI)
    ${GUROBI_INCLUDES}
  endif()
)

file(GLOB SOURCES "src/*.cpp")
add_executable(boosting ${SOURCES})
target_link_libraries(boosting ${RMA_LIBRARIES})
target_link_libraries(boosting ${PEBBL_LIBRARIES})
target_link_libraries(boosting ${Coin_LIBRARIES})

if(ENABLE_GUROBI)
  target_link_libraries(boosting ${GUROBI_LIBRARIES})
endif()

# add_dependencies(boosting coin_external)

# find_package(PythonInterp REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})

# add_subdirectory(pybind11)
# pybind11_add_module(example ./src/python.cpp)
