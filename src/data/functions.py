import requests
from bs4 import BeautifulSoup
import json
import datetime
import numpy as np
import os
from pandas import read_csv

from helpers import isOnOrAfter, formatDate

def scrape(datapath: os.PathLike, link: str, start_date: list = [0,0,0]) -> None:
    """
    Scrapes Powerball drawing data. 
    """
    startyear = start_date[2]
    year = datetime.date.today().year

    RAW = "B1,B2,B3,B4,B5,SB\n" # may need to make adaptable in the future

    for i in range(startyear, year+1):
        url = f"{link}{i}"
        r = requests.get(url)
        print(url)
        data = BeautifulSoup(r.text, 'html.parser')

        datesData, drawingsData = data.select(".col.s_3_12"), data.select(".col.s_9_12") # parallel
        datesData.reverse()
        drawingsData.reverse()

        for j in range(len(datesData)):
            if isOnOrAfter(formatDate(datesData[j].text), start_date):
                draw = BeautifulSoup(str(drawingsData[j]), 'html.parser')
                mainballs = draw.select(".white.ball")

                if "powerball" in link:
                    specialball = int(draw.select(".red.ball")[0].text.replace('Powerball', ''))
                elif "megamillions" in link:
                    specialball = int(draw.select(".gold.ball")[0].text.replace('MegaBall', ''))

                mb = []
                for k in range(len(mainballs)):
                    mb.append(int(mainballs[k].text))

                RAW += ",".join(map(lambda n: f"{n :>2}", mb)) + f",{specialball :>2}\n"

    os.makedirs(datapath, exist_ok=True)

    with open(datapath / "drawings.csv", "w") as outfile:
        outfile.write(RAW)


def generateHistDict(datapath: os.PathLike, WHITEBALLS: int, REDBALLS: int) -> None:
    data = read_csv(datapath / "drawings.csv").to_numpy()
    mainballs = data[:,:5]
    megaballs = data[:,-1]
    
    mballs = {}
    Sballs = {}

    for i in range(1, WHITEBALLS+1):
        mballs[str(i)] = np.sum(mainballs==i, axis=1).tolist()

    for i in range(1, REDBALLS+1):
        Sballs[str(i)] = ((megaballs==i) * 1).tolist()

    history = {
        "mainballs": mballs,
        "specialballs": Sballs
    }

    os.makedirs(datapath, exist_ok=True)

    with open(datapath / "History.json", "w") as outfile:
        hist = json.dumps(history, indent=4).replace('\n            ',  '').replace('\n        ]', ']')
        outfile.write(hist)


def generateFreqDict(datapath: os.PathLike, WHITEBALLS: int, REDBALLS: int) -> None:
    data = read_csv(datapath / "drawings.csv").to_numpy()
    mainballs = data[:,:5]
    megaballs = data[:,-1]
    
    mballs = {}
    Sballs = {}
    draws = data.shape[0]

    for i in range(1, WHITEBALLS+1):
        mballs[str(i)] = int(np.sum(mainballs==i))

    for i in range(1, REDBALLS+1):
        Sballs[str(i)] = int(np.sum(megaballs==i))

    frequency = {
        "mainballs": mballs,
        "specialballs": Sballs,
        "numdraws": draws
    }

    os.makedirs(datapath, exist_ok=True)

    with open(datapath / "Frequencies.json", "w") as outfile:
        json.dump(frequency, outfile, indent=4)

