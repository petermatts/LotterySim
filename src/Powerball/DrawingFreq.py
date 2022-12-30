import requests
from bs4 import BeautifulSoup
import json

def scrape():
    r = requests.get('https://www.powerball.net/statistics')
    s = BeautifulSoup(r.text, 'html.parser')

    # NOTE: only works in _h tag for current data (2015-present)
    # WARNING if ball pool size changes, this will need to be changed 
    h = s.find_all(id='_h')[0]

    data = BeautifulSoup(str(h), 'html.parser').select(".js-stats-selector")
    drawcount = BeautifulSoup(str(h), 'html.parser').select('.draw-count')[0].text
    # print(drawcount)

    # data = [main balls, power balls, power plays]
    mainballData = BeautifulSoup(str(data[0]), 'html.parser').select(".freq-result.js-stats-item")
    powerballData = BeautifulSoup(str(data[1]), 'html.parser').select(".freq-result.js-stats-item")
    powerplayData = BeautifulSoup(str(data[2]), 'html.parser').select(".freq-result.js-stats-item")

    MB = {}
    PB = {}
    PP = {}

    for i in mainballData:
        ball = BeautifulSoup(str(i), 'html.parser').select(".ball.inline")[0].text
        freq = BeautifulSoup(str(i), 'html.parser').strong.text
        MB[ball] = freq.split(' ')[1]

    for i in powerballData:
        ball = BeautifulSoup(str(i), 'html.parser').select(".powerball.inline")[0].text
        freq = BeautifulSoup(str(i), 'html.parser').strong.text
        PB[ball] = freq.split(' ')[1]

    for i in powerplayData:
        play = BeautifulSoup(str(i), 'html.parser').select(".power-play.inline")[0].text
        freq = BeautifulSoup(str(i), 'html.parser').strong.text
        PP[play] = freq.split(' ')[1]

    # print(MB)
    # print(PB)
    # print(PP) 
    # print(drawcount)

    return {
        'mainballs': MB,
        'powerballs': PB,
        'powerplays': PP,
        'numdraws': drawcount
    }

if __name__ == '__main__':
    data = scrape()

    with open("Frequencies.json", "w") as outfile:
        json.dump(data, outfile, indent=4)