import json
import os
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split


def makeDataCSV(type_: str, path: str):
    if type_.lower() == "powerball":
        historyfile = "../Powerball/History.json"
        SB = 'powerballs'
    elif type_.lower() == "megamillions":
        historyfile = "../MegaMillions/History.json"
        SB = 'megaballs'
    else:
        raise ValueError("Invalid Lottery type")
    

    with open(historyfile) as f:
        data = json.load(f)

    output = "B1,B2,B3,B4,B5,SB\n"

    total = len(data["mainballs"][0])

    for i in range(total):
        line = ""

        for j in range(len(data["mainballs"])):
            if data["mainballs"][j][i] == 1:
                if j < 9:
                    line += ' ' + str(j+1) + ','
                else:
                    line += str(j+1) + ','

        for j in range(len(data[SB])):
            if data[SB][j][i] == 1:
                if j < 9:
                    line += ' ' + str(j+1) + '\n'
                else:
                    line += str(j+1) + '\n'
                break
        output += line

    with open("%s%s_drawings.csv"%(path, type_), 'w') as f:
        f.writelines(output)


def generateTrainTestData(lottery_type: str, savepath: str, savefile: str, lookback: int, validate: float = 0.1, test: float = 0.1, randTest: int = 100):
    """
    @param historyfile (required) the history file of the lottery the model is being trained on ('powerball' or 'megamillions')
    @param savepath    (required) relative or absolute path to save data
    @param savefile    (required) name of datafile to save must be a .npy or .npz file
    @param lookback    (required) number of datapoints to look back when collecting data sequentially
    @param validate    (optional) float variable indicating the percentage of train data to be used for validation\
                       default: 0.1
    @param test        (optional) float variable indicating the percentage of data to be used for testing\
                       default: 0.1
    @param randTest    (optional) integer variable specifying the size of the randomly generated test data\
                       Only used if test is 0, default: 100
    """

    if not savepath.endswith('/'):
        savepath += '/'

    if lottery_type.lower() == "powerball":
        num_mainballs = 69
        num_specialballs = 26
        freqpath = "../Powerball/Frequencies.json"
        SB = 'powerballs'
    elif lottery_type.lower() == "megamillions":
        num_mainballs = 70
        num_specialballs = 25
        freqpath = "../MegaMillions/Frequencies.json"
        SB = 'megaballs'
    else:
        raise ValueError("Invalid Lottery type")

    if not os.path.isfile("%s%s_drawings.csv"%(savepath, lottery_type.lower())):
        makeDataCSV(lottery_type, savepath)

    raw_data = read_csv("%s%s_drawings.csv"%(savepath, lottery_type.lower())).to_numpy()
    
    mainballs_x = []
    mainballs_y = []
    specialballs_x = []
    specialballs_y = []

    for i in range(raw_data.shape[0] - lookback):
        mainball_x = raw_data[i:i+lookback,:-1]
        mainball_y = np.zeros(num_mainballs)
        for i in raw_data[i+lookback,:-1]:
            mainball_y[i-1] = 1
        mainballs_x.append(mainball_x)
        mainballs_y.append(mainball_y)

        powerball_x = raw_data[i:i+lookback,:-1]
        powerball_y = np.zeros(num_specialballs)
        powerball_y[raw_data[i+lookback, -1] - 1] = 1
        specialballs_x.append(powerball_x)
        specialballs_y.append(powerball_y)
    
    if test != 0:
        mainballs_xtr, mainballs_xte, mainballs_ytr, mainballs_yte = train_test_split(mainballs_x, mainballs_y, test_size=test)
        specialballs_xtr, specialballs_xte, specialballs_ytr, specialballs_yte = train_test_split(specialballs_x, specialballs_y, test_size=test)
    else:
        mainballs_xtr = mainballs_x
        mainballs_ytr = mainballs_y
        specialballs_xtr = specialballs_x
        specialballs_ytr = specialballs_y

        mainballs_xte = []
        mainballs_yte = []
        specialballs_xte = []
        specialballs_yte = []

        with open(freqpath) as f:
            freq = json.load(f)

        mainball_range = list(map(lambda x: int(x), freq["mainballs"].keys()))
        powerball_range = list(map(lambda x: int(x), freq[SB].keys()))

        mainball_probs = list(map(lambda x: x/freq["numdraws"], freq["mainballs"].values()))
        powerball_probs = list(map(lambda x: x/freq["numdraws"], freq[SB].values()))

        for t in range(randTest):
            mainball_instance = list()
            powerball_instance = list()

            for i in range(lookback):
                mbi = np.random.choice(mainball_range, size=5, replace=False, p=mainball_probs).tolist()
                mbi.sort()
                mainball_instance.append(mbi)
                powerball_instance.append(np.random.choice(powerball_range, replace=False, p=powerball_probs))

            mainballs_xte.append(mainball_instance)
            specialballs_xte.append(powerball_instance)

            m_y = np.zeros(num_mainballs)
            s_y = np.zeros(num_specialballs)
            m_y[np.random.choice(mainball_range, mainball_probs)] = 1
            s_y[np.random.choice(powerball_range, powerball_probs)] = 1
            mainballs_yte.append(m_y)
            specialballs_yte.append(s_y)


    mainballs_xtr, mainballs_xval, mainballs_ytr, mainballs_yval = train_test_split(mainballs_xtr, mainballs_ytr, test_size=validate)
    specialballs_xtr, specialballs_xval, specialballs_ytr, specialballs_yval = train_test_split(specialballs_xtr, specialballs_ytr, test_size=validate)

    np.savez_compressed(
        savepath+savefile,
        mainballs_xtr=mainballs_xtr,
        mainballs_xte=mainballs_xte,
        mainballs_ytr=mainballs_ytr,
        mainballs_yte=mainballs_yte,
        mainballs_xval=mainballs_xval,
        mainballs_yval=mainballs_yval,
        specialballs_xtr=specialballs_xtr,
        specialballs_xte=specialballs_xte,
        specialballs_ytr=specialballs_ytr,
        specialballs_yte=specialballs_yte,
        specialballs_xval=specialballs_xval,
        specialballs_yval=specialballs_yval
    )

if __name__ == "__main__":
    generateTrainTestData("../Powerball/History.json", "Data/Train10.npz", 10) #Powerball dataset with lookback 10
