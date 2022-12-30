import requests
from bs4 import BeautifulSoup
import json
import datetime

# NOTE powerball ball pool changed on 10/7/2015
# all scraping is from the drawing on that day to present
def scrape():
    startDate = [10, 7, 2015]
    startyear = startDate[2]
    year = datetime.date.today().year
    # print(year)

    MB = [[]]*69
    PB = [[]]*26

    dates = [] 
    for i in range(startyear, year+1):
        r = requests.get('https://www.lottodatabase.com/lotto-database/american-lotteries/powerball/draw-history/'+str(i))
        print('https://www.lottodatabase.com/lotto-database/american-lotteries/powerball/draw-history/'+str(i))
        data = BeautifulSoup(r.text, 'html.parser')

        datesData = data.select(".col.s_3_12") # parallel
        drawingsData = data.select(".col.s_9_12") # parallel
        datesData.reverse()
        drawingsData.reverse()

        # print(drawingsData)

        # print(len(drawings['mainballs']))
        # print(len(drawings['powerballs']))


        for j in range(len(datesData)):
            dates.append(formatDate(datesData[j].text))
            if isOnOrAfter(formatDate(datesData[j].text), startDate):

                draw = BeautifulSoup(str(drawingsData[j]), 'html.parser')
                mainballs = draw.select(".white.ball")
                powerball = int(draw.select(".red.ball")[0].text.replace('Powerball', ''))

                mb = []
                for k in range(len(mainballs)):
                    mb.append(int(mainballs[k].text))
                # print(mb)

                # print(powerball)

                for m in range(1, len(MB)+1):
                    hist = list(MB[m-1])
                    # hist = list(MB[str(m)])
                    if m in mb:
                        hist.append(1)
                        MB[m-1] = hist
                    else:
                        hist.append(0)
                        MB[m-1] = hist
                    # MB[str(m)] = hist

                for p in range(1, len(PB)+1):
                    hist = list(PB[p-1])
                    # hist = list(PB[str(p)])
                    if p == powerball:
                        hist.append(1)
                        PB[p-1] = hist
                    else:
                        hist.append(0)
                        PB[p-1] = hist
                    # PB[str(p)] = hist

        # print(MB)
        # print(PB)

    return {
        'mainballs': MB,
        'powerballs': PB
    }, dates


def monthToNum(month: str):
    m = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }
    return m[month]


# Input: string date "${Day}, ${Month} ${NumberDay}, ${Year}"
# Output: [Month, Day, Year]
def formatDate(date: str):
    split = date.split(', ')
    day = int(split[1].split(' ')[1])
    month = monthToNum(split[1].split(' ')[0])
    year = int(split[2])
    
    return [month, day, year]

# NOTE expects input that came from formateDate [month, day, year]
# returns true if date1 is the same as or after date2
def isOnOrAfter(date1: list, date2: list):
    if date1[2] < date2[2]: # year1 < year2
        return False
    elif date1[2] > date2[2]: # year1 > year2
        return True
    else:
        if date1[0] < date2[0]: # month1 < month2
            return False
        elif date1[0] > date2[0]: # month1 > month2
            return True
        else:
            if date1[1] < date2[1]: # day1 < day2
                return False
            else:
                return True


def printBalls(data: list, title: str):
    print('\n'+title+'\n')
    for i in range(len(data)):
        print(i+1, data[i])

def DetailedPrint(drawings: dict):
    printBalls(drawings['powerballs'], 'Powerballs')
    printBalls(drawings['mainballs'], 'Mainballs')

if __name__ == '__main__':
    data, dates = scrape()

    # DetailedPrint(data)

    # print(dates)

    # print(isOnOrAfter([10,6,2022],[10,7,2015]))

    with open("History.json", "w") as outfile:
        json.dump(data, outfile, indent=4)