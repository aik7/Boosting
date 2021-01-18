# Boosting

[![Build Status](https://travis-ci.com/aik7/Boosting.svg?branch=devel)](https://travis-ci.com/aik7/Boosting)

We implemented both classification and regression algorithms using rectangular maximum agreement problem (RMA) as a subproblem.

## LPBR (LPBoost with RMA)

LPBR is a two-class classification algorithm using LPBoost and RMA. (currently not available with CLP)

## REPR (Rule-Enhanced Penalized Regression)

REPR is a prediction algorithm using linear regression with both linear and boxed-based rule variables.

## Software Requirement:

### You need to install
* CMake (version >= 3.0)
* C++ compiler (g++)
* OpenMPI 2.1.1 (openmpi-bin, libopenmpi-dev)
* Fortran compiler (gfortran)
* BLAS and LAPACK packages (libblas-dev, liblapack-dev)

### The following packages are installed by running scripts/build.sh as described below
* [PEBBL](https://github.com/PEBBL/pebbl)
* [RMA](https://github.com/aik7/RMA)
* [Coin-OR/CLP](https://github.com/coin-or/Clp)

### Optional packages
* (Optional): [Gurobi](http://www.gurobi.com/)

### Note
* The build was tested on Ubuntu 18.04 (Bionic) as shown in [our TravisCI file](https://github.com/aik7/Boosting/blob/devel/.travis.yml)

## The description and user guide of Boosting algorithms
* [Presentation](https://github.com/aik7/Boosting/blob/master/Boosting.pdf)
* User Guide
* Information about the [RMA](https://github.com/aik7/RMA) sub-package

## How to download and build Boosting algorithm

* Clone or download this Boosting repository
```
git clone --recursive https://github.com/aik7/Boosting.git
```
* Run the following command in the Boosting main directory
  to build Boosting along with PEBBL, RMA, and Coin-OR CLP
```
sh scripts/build.sh
```

### Compile with Gurobi

#### Option 1:
* If you want to download Gurobi and compile with Boosting,
  you can run the following command in the Boosting main directory
  to build Boosting along with PEBBL, RMA, Coin-OR CLP and Gurobi.
```
sh scripts/build_gurobi.sh
```

#### Option 2:

* If you already have Gurobi Linux 9.1 version, run the following commands in the Boosting main directory by setting your GUROBI home directory `<gurobi_home_dir>`:
  ```
  mkdir build ; cd build; cmake -DENABLE_GUROBI=true -DGUROBI_HOME=<gurobi_home_dir> .. ; make
  ```

* If your Gurobi version is different, read [How to compile with Gurobi](https://github.com/aik7/Boosting/wiki/How-to-compile-with-Gurobi)


#### Please set your Gurobi license


## Example run commands:

### Serial implementation
```
./build/boosting <data_filename>
```

* You can use a sample data, `./data/servo.data` for `<data_filename>`.

### Parallel implementation
```
mpirun -np 4 ./build/boosting <data_filename>
```

### Using parameters
```
./build/boosting --numIterations=10 --isUseGurobi=true --c=0.5 --e=0.5 --p=2 <data_filename>
```

* You can set the number of boosting iterations using `numIterations`.

## Class Diagram

<p align="center">

<img src="https://github.com/aik7/Boosting/blob/devel/figures/Boosting_class_org.png" width="400">

* A solid arrow indicates an inheritance relationship
* A dashed arrow indicates a composition relationship

## Source files at src directory

```
├── argBoost.cpp     : a file contains Boosting argument class
├── argBoost.h
├── argBoost.o
├── boosting.cpp     : a file contains Boosting driver class
├── boosting.h
├── driver.cpp       : a driver file
├── lpbr.cpp         : a file contains LPBR class
├── lpbr.h
├── repr.cpp         : a file contains REPR class
└── repr.h
```

## Reference

```
@article{doi:10.1287/ijoo.2019.0015,
  author = {Eckstein, Jonathan and Kagawa, Ai and Goldberg, Noam},
  title = {REPR: Rule-Enhanced Penalized Regression},
  journal = {INFORMS Journal on Optimization},
  volume = {1},
  number = {2},
  pages = {143-163},
  year = {2019}
}
```

```
@phdthesis{AiThesis,
  author       = {Ai Kagawa},
  title        = {The Rectangular Maximum Agreement Problem: Applications and Parallel Solution},
  school       = {Rutgers University},
  year         = 2018
}
```

```
@InProceedings{egk2017,
  title     =  {Rule-Enhanced Penalized Regression by Column Generation using Rectangular Maximum Agreement},
  author    =  {Jonathan Eckstein and Noam Goldberg and Ai Kagawa},
  booktitle =  {Proceedings of the 34th International Conference on Machine Learning},
  pages     =  {1059--1067},
  year      =  {2017},
  volume    =  {70},
  series    =  {Proceedings of Machine Learning Research},
  address   =  {Sydney, Australia}
}
```
