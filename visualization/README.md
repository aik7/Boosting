# Visualization of Boosting Results

* Always run the following command in the Boosting main directory
* All plots are saved in the `visualization/plots` directory unless you set `-output_dir`
* Specify the input file name `<prediction*.out>` or `<error*.out>`

## Plot the prediction vs actual y-values

```
python visualization/src/generatePlot.py -isPrediction -input_file ./results/<prediction*.out>
```

## Plot the errors vs iterations

```
python visualization/src/generatePlot.py -isError -input_file ./results/<error*.out>
```
