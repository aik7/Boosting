# Compute predicted y-values using the trained model

## How to build

* Run the following command at the current directory to build

```
sh build.sh
```

## How to run

* The fist file is the saved REPR model (i.e. indian_model_*.out)
* The second file is the testing X-values (i.e. indian.dat)

```
./predict ../indian_model_*.out  ../data/indian.dat
```
