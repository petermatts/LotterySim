import argparse
import os
from datetime import datetime
from DataGen import *

def configName(args: argparse.Namespace) -> str:
    """Returns a string representation of the model configuration, and current date"""
    if args.modelname is None:
        now = datetime.now()
        day = str(now.day).rjust(2, '0')
        month = str(now.month).rjust(2, '0')
        year = str(now.year)

        if not args.powerball and not args.megamillions:
            raise ValueError("Must specify a Lottery Type")
        if not args.mainball and not args.specialball:
            raise ValueError("Must specify a Ball Type")
        if not args.RNN and not args.LSTM and not args.Transformer:
            raise ValueError("Must specify a Model Type")

        lotto = "PB" if args.powerball is not None else "MM"
        ball = "MB" if args.mainball is not None else "SB"
        if args.RNN:
            modeltype = "RNN_"
        elif args.LSTM:
            modeltype = "LSTM"
        elif args.Transformer:
            modeltype = "TRFM"

        return "%s-%s-%s_TR%0.2dVL%0.2dTE%0.2d_%s-%s-%s"%(lotto, ball, modeltype, args.split[0], args.split[1], args.split[2], month, day, year)
    else:
        return args.modelname


def train():
    pass # function to run training

def validate():
    pass # function to perform validation

def test():
    pass # function to perform testing

def makeModel(args: argparse.Namespace):
    # construct model definition and call train/validate/test
    # create model folder for saving data and model config.yaml or specs.yaml (havent decided on a name yet)
    if args.evaluate:
        pass
    else:
        if not os.path.exists("Models"):
            os.mkdir("Models")

        modelname = args.modelname if args.modelname is not None else configName(args)
        os.mkdir("Models/%s"%modelname)
        os.mkdir("Models/%s/Data"%modelname)

        # Make the data for the model
        assert args.lookback is not None
        datapath = "Models/%s/Data/"%modelname
        datafile = "data%d.npz"%args.lookback
        if args.powerball:
            generateTrainTestData('powerball', datapath, datafile, args.lookback, args.split[1], args.split[2])
        elif args.megamillions:
            generateTrainTestData('megamillions', datapath, datafile, args.lookback, args.split[1], args.split[2])
        else:
            raise ValueError("Must specify a Lottery Type")

        #todo keep going with model setup





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('--modelname', type=str, help="Name of the model, will be saved using this name so be specific")
    parser.add_argument("-e", "--evaluate", type=bool, nargs='?', const=True, default=False, help="Activate test mode - Evaluate model on val/test set (no training)")    
    
    modeltype = parser.add_mutually_exclusive_group()
    modeltype.add_argument('--RNN', nargs='?', type=bool, const=True, default=False, help="Train RNN model")
    modeltype.add_argument('--LSTM', nargs='?', type=bool, const=True, default=False, help="Train LSTM model")
    modeltype.add_argument('--Transformer', nargs='?', type=bool, const=True, default=False, help="Train Transformer model")


    # lottery type args
    lotto = parser.add_mutually_exclusive_group()
    lotto.add_argument('-mm', '--megamillions', nargs='?', type=bool, const=True, default=False, help="Selects model over the MegaMillions Lottery data")
    lotto.add_argument('-pb', '--powerball', nargs='?', type=bool, const=True, default=False, help="Selects model over the Powerball Lottery data")

    # ball type args
    ball = parser.add_mutually_exclusive_group()
    ball.add_argument('-mb', '--mainball', type=bool, nargs='?', const=True, default=False, help="Selects model over the specified lottery main balls")
    ball.add_argument('-sb', '--specialball', type=bool, nargs='?', const=True, default=False, help="Selects model over the specified lottery special balls")

    # data args
    parser.add_argument('--split', nargs=3, type=int, default=[80,10,10], help="3 int inputs, for training, validation, and test split percentages in that order. Must sum to 100. If unspecified default will be 80,10,10") #?
    parser.add_argument('--lookback', type=int, default = 10, help="The number of datapoints to lookback in a time series")
    parser.add_argument('--randtest', type=int, nargs='?', default=100, help="Number of random test data points to make, only used if test split is 0")
    
    # model args/hyperparameter args #todo




    ################################################################################################
    args = parser.parse_args()
    # print(args)
    # print(configName(args))
    makeModel(args)
