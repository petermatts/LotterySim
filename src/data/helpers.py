"""
Helpers.py

This file includes helperfunctions to help with the data collection algorithms
"""

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


def formatDate(date: str):
    """
    Input: string date f"{Day}, {Month} {NumberDay}, {Year}"
    Output: [Month, Day, Year]
    """
    split = date.split(', ')
    day = int(split[1].split(' ')[1])
    month = monthToNum(split[1].split(' ')[0])
    year = int(split[2])
    
    return [month, day, year]


def isOnOrAfter(date1: list, date2: list):
    """
    Input: date in list form [month, day, year]
    Output: true if date1 is the same as or after date2
    """
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
                return True # date1 == date2


def printBalls(data: list, title: str):
    print('\n'+title+'\n')
    for i, v in enumerate(data):
        print(f"{i+1} {v}")

def DetailedPrint(drawings: dict, name1: str, name2: str):
    printBalls(drawings['megaballs'], name1)
    printBalls(drawings['mainballs'], name2)
