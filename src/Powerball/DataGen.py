import json
import os
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split


def makeDataCSV():
    with open("History.json") as f:
        data = json.load(f)

    output = "B1,B2,B3,B4,B5,PB\n"

    total = len(data["mainballs"][0])

    for i in range(total):
        line = ""

        for j in range(len(data["mainballs"])):
            if data["mainballs"][j][i] == 1:
                if j < 9:
                    line += ' ' + str(j+1) + ','
                else:
                    line += str(j+1) + ','

        for j in range(len(data["powerballs"])):
            if data["powerballs"][j][i] == 1:
                if j < 9:
                    line += ' ' + str(j+1) + '\n'
                else:
                    line += str(j+1) + '\n'
                break
        output += line

    if not os.path.exists("Data"):
        os.mkdir("Data")

    with open("Data/drawings.csv", 'w') as f:
        f.writelines(output)


def generateTrainTestData(savefile: str, lookback: int, validate: float = 0.2, test: float = 0.1, randTest: int = 100):
    """
    @param savefile (required) relative or absolute path to save data, must be a .npy or .npz file
    @param lookback (required) number of datapoints to look back when collecting data sequentially
    @param validate (optional) float variable indicating the percentage of train data to be used for validation
                    default: 0.2
    @param test     (optional) float variable indicating the percentage of data to be used for testing
                    default: 0.1
    @param randTest (optional) integer variable specifying the size of the randomly generated test data
                    Only used if test is 0, default: 100
    """

    if not os.path.isfile("Data/drawings.csv"):
        makeDataCSV()

    raw_data = read_csv("Data/drawings.csv").to_numpy()
    
    mainballs_x = []
    mainballs_y = []
    powerballs_x = []
    powerballs_y = []


    for i in range(raw_data.shape[0] - lookback):
        mainball_x = raw_data[i:i+lookback,:-1]
        mainball_y = np.zeros(69)
        for i in raw_data[i+lookback,:-1]:
            mainball_y[i-1] = 1
        mainballs_x.append(mainball_x)
        mainballs_y.append(mainball_y)

        powerball_x = raw_data[i:i+lookback,:-1]
        powerball_y = np.zeros(26)
        powerball_y[raw_data[i+lookback, -1] - 1] = 1
        powerballs_x.append(powerball_x)
        powerballs_y.append(powerball_y)

    data = dict()
    
    mainballs_xtr, mainballs_xte, mainballs_ytr, mainballs_yte = train_test_split(mainballs_x, mainballs_y, test_size=test)
    mainballs_xtr, mainballs_xval, mainballs_ytr, mainballs_yval = train_test_split(mainballs_xtr, mainballs_ytr, test_size=validate)

    powerballs_xtr, powerballs_xte, powerballs_ytr, powerballs_yte = train_test_split(powerballs_x, powerballs_y, test_size=test)
    powerballs_xtr, powerballs_xval, powerballs_ytr, powerballs_yval = train_test_split(powerballs_xtr, powerballs_ytr, test_size=validate)

    np.savez_compressed(savefile,
        mainballs_xtr=mainballs_xtr,
        mainballs_xte=mainballs_xte,
        mainballs_ytr=mainballs_ytr,
        mainballs_yte=mainballs_yte,
        mainballs_xval=mainballs_xval,
        mainballs_yval=mainballs_yval,
        powerballs_xtr=powerballs_xtr,
        powerballs_xte=powerballs_xte,
        powerballs_ytr=powerballs_ytr,
        powerballs_yte=powerballs_yte,
        powerballs_xval=powerballs_xval,
        powerballs_yval=powerballs_yval
    )

if __name__ == "__main__":
    generateTrainTestData("Data/Train10.npz", 10)
