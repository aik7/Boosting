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

* Run the following command in the Boosting main directory to run REPR using a training dataset
```
./build/boosting <train_data_filename>
```

* You can use a sample data, `./data/servo.data` for `<train_data_filename>`.

* If you want to test REPR for both the train and test datasets
```
./build/boosting <train_data_filename> <test_data_filename>
```

* The test dataset is an optional, but you have to have the train dataset.

### Parallel implementation
```
mpirun -np 4 ./build/boosting <train_data_filename>
```

### Parameters

| parameters      |      description                               | data type | range         | default value  |
|-----------------|:-----------------------------------------------|:---------:|--------------:|---------------:|
| numIterations   | the number of boosting iterations              | integer   | [0, infinity) | 1              |
| isUseGurobi     | Use Gurobi instead of CLP to solve the restricted master Problem (RMP). If you want to enable this option, you have to compile with Gurobi. | bool      | true or false | false          |
| p               | the exponent of each observation's error variable in RMP | integer    | 1 or 2       | 1 for CLP; 2 for Gurobi |
| c               | a penalty term for linear coefficients in RMP | double     | [0, infinity) | 1.0           |
| e               | a penalty term for rule coefficients in RMP | double     | [0, infinity) | 1.0           |
| isEvalEachIter  | whether or not to evaluate the current REPR model using MSE in each boosting iteration | bool      | true or false | true           |
| isSavePredictions  | whether or not to save the actual and REPR predicted response values in a file after the training  | bool      | true or false | true           |
| isSaveAllRMASols  | whether or not to save the Greedy and PEBBL RMA solutions of each boosting iteration in a file   | bool      | true or false | false          |
| isSaveWts      | whether or not to save the weights of each boosting iteration in a file      | bool      | true or false | false          |


* The following is an example command to run REPR using the parameters.
```
./build/boosting --numIterations=10 --isUseGurobi=true --c=0.5 --e=0.5 --p=2 <data_filename>
```

## Class Diagram

<p align="center">

<img src="https://github.com/aik7/Boosting/blob/devel/figures/Boosting_class_org.png" width="400">

* A solid arrow indicates an inheritance relationship
* A dashed arrow indicates a composition relationship

## Source files at src directory

```
├── argBoost.cpp     : a file contains Boosting argument class
├── argBoost.h
├── boosting.cpp     : a file contains Boosting class
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
