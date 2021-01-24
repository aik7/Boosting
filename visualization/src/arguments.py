import argparse


# ******************* get all arguments *******************
def parse_args():

    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('-isPrediction', action='store_true',
                        help='generate the prediction plot.', required=False)

    parser.add_argument('-isError', action='store_true',
                        help='generate the error plot.', required=False)

    parser.add_argument('-isPrintInfo', action='store_true',
                        help='print info or not.', required=False)

    parser.add_argument('-isSortActY', action='store_true',
                        help='print info or not.', required=False)

    # parser.add_argument('-input_dir', default='results', type=str,
    #                     help='input directory', required=False)

    parser.add_argument('-output_dir', default='visualization/plots', type=str,
                        help='output directory', required=False)

    parser.add_argument('-input_file', default='', type=str,
                        help='input data file', required=False)

    parser.add_argument('-output_file', default='', type=str,
                        help='output data file', required=False)


    return parser.parse_args()
