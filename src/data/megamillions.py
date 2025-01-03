import requests
from bs4 import BeautifulSoup
import json
import datetime
import numpy as np
from pathlib import Path
import os
from pandas import read_csv

from helpers import isOnOrAfter, formatDate

DATAPATH = (Path(__file__).parent / "../../Data/MegaMillions").resolve()
WHITEBALLS = 70
REDBALLS = 25

def scrape() -> None:
    """
    Scrapes Megamillions drawing data. 
    NOTE megamillions ball pool changed on 10/31/2017, all scraping is from the drawing on that day to present
    """
    startDate = [10, 31, 2017]
    startyear = startDate[2]
    year = datetime.date.today().year

    RAW = "B1,B2,B3,B4,B5,MB\n"

    for i in range(startyear, year+1):
        r = requests.get('https://www.lottodatabase.com/lotto-database/american-lotteries/megamillions/draw-history/'+str(i))
        print('https://www.lottodatabase.com/lotto-database/american-lotteries/megamillions/draw-history/'+str(i))
        data = BeautifulSoup(r.text, 'html.parser')

        datesData, drawingsData = data.select(".col.s_3_12"), data.select(".col.s_9_12") # parallel
        datesData.reverse()
        drawingsData.reverse()

        for j in range(len(datesData)):
            if isOnOrAfter(formatDate(datesData[j].text), startDate):
                draw = BeautifulSoup(str(drawingsData[j]), 'html.parser')
                mainballs = draw.select(".white.ball")
                megaball = int(draw.select(".gold.ball")[0].text.replace('MegaBall', ''))

                mb = []
                for k in range(len(mainballs)):
                    mb.append(int(mainballs[k].text))

                RAW += ",".join(map(lambda n: f"{n :>2}", mb)) + f",{megaball :>2}\n"

    os.makedirs(DATAPATH, exist_ok=True)

    with open(DATAPATH / "drawings.csv", "w") as outfile:
        outfile.write(RAW)


def generateHistDict() -> None:
    data = read_csv(DATAPATH / "drawings.csv").to_numpy()
    mainballs = data[:,:5]
    megaballs = data[:,-1]
    
    mballs = {}
    Mballs = {}

    for i in range(1, WHITEBALLS+1):
        mballs[str(i)] = np.sum(mainballs==i, axis=1).tolist()

    for i in range(1, REDBALLS+1):
        Mballs[str(i)] = ((megaballs==i) * 1).tolist()

    history = {
        "mainballs": mballs,
        "megaballs": Mballs
    }

    os.makedirs(DATAPATH, exist_ok=True)

    with open(DATAPATH / "History.json", "w") as outfile:
        hist = json.dumps(history, indent=4).replace('\n            ',  '').replace('\n        ]', ']')
        outfile.write(hist)


def generateFreqDict() -> None:
    data = read_csv(DATAPATH / "drawings.csv").to_numpy()
    mainballs = data[:,:5]
    megaballs = data[:,-1]
    
    mballs = {}
    Mballs = {}
    draws = data.shape[0]

    for i in range(1, WHITEBALLS+1):
        mballs[str(i)] = int(np.sum(mainballs==i))

    for i in range(1, REDBALLS+1):
        Mballs[str(i)] = int(np.sum(megaballs==i))

    frequency = {
        "mainballs": mballs,
        "megaballs": Mballs,
        "numdraws": draws
    }

    os.makedirs(DATAPATH, exist_ok=True)

    with open(DATAPATH / "Frequencies.json", "w") as outfile:
        json.dump(frequency, outfile, indent=4)
