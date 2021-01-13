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

* After training a REPR model for a given dataset, the REPR code saves a trained REPR model as `[datafile_name]_model*.out`.  The following is an example of a saved model with some comments.

```
#_of_attributes: 10  // # of linear variables (# of attributes)
#_of_boxes:      2   // # of boxes/rules

bias: 1.65164

coefficients_for_linear_variables:
0.0495856 0.0396685 0.0360222 -0.106028 0.130792 0.029447 7.77962e-13 2.4756e-09 -0.183531 0.24702

coefficients_for_box_variables:
-2.05791 -0.277335

the_average_value_of_y_value: 1.38971

the_standard_deviation_of_y_value: 1.55496

the_average_value_of_each_attribute:
0.215569 0.215569 0.239521 0.131737 0.251497 0.209581 0.185629 0.179641 4.15569 2.53892

the_standard_deviation_of_each_attribute:
0.411216 0.411216 0.426791 0.338204 0.433874 0.407009 0.388807 0.383888 1.01472 1.36574

Box_0_a: -inf -inf -inf -inf -inf -inf -inf -inf 4 -inf   // lower bounds (from attribute 0 to n) of box 0
Box_0_b: inf inf inf inf inf inf inf inf inf inf          // upper bounds (from attribute 0 to n) of box 0
Box_1_a: -inf -inf -inf -inf -inf -inf -inf -inf 5 2      // lower bounds (from attribute 0 to n) of box 1
Box_1_b: inf inf inf inf inf inf inf inf inf inf          // upper bounds (from attribute 0 to n) of box 1

```
