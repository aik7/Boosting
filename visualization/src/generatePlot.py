import os, sys
from   arguments import parse_args
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})


def plotPredAct(pred_y, act_y, title="Prediction",
                output_dir="", output_file="prediction.pdf", y_unit=""):

    # errors = np.array(errors)

    plt.figure(figsize=(5, 5))
    plt.style.use('seaborn-darkgrid')

    plt.subplot(1, 1, 1) # nrows, ncols, index

    plt.plot(pred_y, 'x', label='Prediction')
    plt.plot(act_y,  '.', label='Actual')

    plt.title(title)

    plt.xlabel('Observations')
    plt.ylabel(y_unit) #, rotation='horizontal')

    plt.legend()
    plt.tight_layout()

    filename = os.path.join(output_dir, output_file)
    plt.savefig(filename, pad_inches=0.2)

    print("Saved file: ", filename)


def generatePredictionPlot(parser):

    filename = parser.input_file
    # filename ="results/predictionTrain_data.txt.out" #"predictionTrain.servo.data"

    df = pandas.read_csv(filename, sep='\t')

    if parser.isPrintInfo:
        print(df)

    predY = df['ActY']
    actY  = df['BoostingPredictedY']

    matData = np.column_stack((predY, actY))

    if (parser.isSortActY):
      matData = matData[matData[:,1].argsort()]

    out_file = "prediction.pdf" if parser.output_file=="" else parser.output_file

    plotPredAct(matData[:,0], matData[:,1],
               output_dir=parser.output_dir,
               output_file=out_file)


def plotError(err_train, err_test=[], title="MSE",
              output_dir="", output_file="error.pdf", y_unit=""):

    # errors = np.array(errors)

    plt.figure(figsize=(5, 5))
    plt.style.use('seaborn-darkgrid')

    plt.subplot(1, 1, 1) # nrows, ncols, index

    plt.plot(err_train, 'x', label='Train')

    if len(err_test)!=0:
        plt.plot(err_test,  '.', label='Test')

    plt.title(title)

    plt.xlabel('Iterations')
    plt.ylabel(y_unit) #, rotation='horizontal')

    plt.legend()
    plt.tight_layout()

    filename = os.path.join(output_dir, output_file)
    plt.savefig(filename, pad_inches=0.2)

    print("Saved file: ", filename)


def generateErrorPlot(parser):

    filename = parser.input_file
    # filename ="results/predictionTrain_data.txt.out" #"predictionTrain.servo.data"

    df = pandas.read_csv(filename, sep='\t')

    if parser.isPrintInfo:
        print(df)

    errTrain = df['Train_MSE']
    errTest  = df['Test_MSE'] if len(df.columns) == 3 else []

    out_file = "error.pdf" if parser.output_file=="" else parser.output_file

    plotError(errTrain, errTest,
              output_dir=parser.output_dir,
              output_file=out_file)


def main():

    parser = parse_args()

    if not os.path.exists(parser.output_dir):
        os.makedirs(parser.output_dir)

    if (parser.isPrediction):
        generatePredictionPlot(parser)

    elif (parser.isError):
        generateErrorPlot(parser)

    else:
        print("You have to choose either isPrediction xor isPlotError to be true!")


if __name__ == "__main__":
    main()
