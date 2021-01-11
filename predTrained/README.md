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
./predict ../servo_model_*.out  ../data/servo.data
```


## Saved REPR model

* After training a REPR model for a given dataset, the REPR code saves a trained REPR model (`[datafile_name]_model*.out`)

```
#_of_attributes: 10   // # of linear variables (# of attributes)
#_of_boxes:      2    // # of boxes/rules

bias: 0.752034

coefficients for linear variables:
0.0495856 0.0396685 0.0360222 -0.106028 0.130792 0.029447 1.25132e-09 1.23471e-09 -0.973404 0.24702

coefficients for box variables:
-0.778416 -0.501081

Box_0_a: 10 : 0 0 0 0 0 0 0 0 4 0  // lower bounds (from attribute 0 to n) of box 0
Box_0_b: 10 : 2 2 2 2 2 2 2 2 5 6  // upper bounds (from attribute 0 to n) of box 0
Box_1_a: 10 : 0 0 0 0 0 0 0 0 4 0  // lower bounds (from attribute 0 to n) of box 1
Box_1_b: 10 : 2 2 2 2 2 2 2 2 4 6  // upper bounds (from attribute 0 to n) of box 1
```
