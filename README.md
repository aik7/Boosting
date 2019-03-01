# Boosting

We implemented both classification and regression algorithms using rectangular maximum agreement problem (RMA) as a subproblem.

## LPBR (LPBoost with RMA)

LPBR is a two-class classification algorithm using LPBoost and RMA.

## REPR (Rule-Enhanced Penalized Regression) 

REPR is a prediction algorithm using linear regression with both linear and boxed-based rule variables.

## Software Requirement:
* C++ compiler
* [PEBBL](https://software.sandia.gov/trac/acro/wiki/Example/Building/acro-pebbl)
* [Gurobi](http://www.gurobi.com/)
* MPI 
* R with RuleFit, randomForest, gbm, and fastAdaboost packages

## The description and user guide of RMA
* Presentation
* User Guide

## How to download and build Boosting algorithm

* Clone or download this Boosting repository
```
git clone https://github.com/aik7/Boosting.git
```
* Run the following command for compiling and building applications
```
cd Boosting
make
```

## Example commands to run RMA:

### Serial implementation
```
./boosting <data_filename>
```

### Parallel implementation
```
mpirun -np 4 ./boosting <data_filename>
```

Please read the user guide about how to use parameters for the Boosting solver.

## Reference

```
@phdthesis{AiThesis,
  author       = {Ai Kagawa}, 
  title        = {The Rectangular Maximum Agreement Problem: Applications and 
                  Parallel Solution},
  school       = {Rutgers University},
  year         = 2018
}
```

```
@InProceedings{egk2017,
  title =    {Rule-Enhanced Penalized Regression by Column Generation using Rectangular
              Maximum Agreement},
  author =   {Jonathan Eckstein and Noam Goldberg and Ai Kagawa},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  pages =    {1059--1067},
  year =   {2017},
  editor =   {D. Precup and Y.W. Teh},
  volume =   {70},
  series =   {Proceedings of Machine Learning Research},
  address =    {Sydney, Australia}
}
```
