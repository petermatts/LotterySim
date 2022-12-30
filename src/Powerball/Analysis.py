import json

def timesSinceLastDraw():
    history = json.load(open("History.json"))

    mb = {}
    pb = {}

    # if history['mainballs'] exits
    x = 1
    for i in history['mainballs']:
        tmp = i
        tmp.reverse()
        mb[str(x)] = tmp.index(1)
        x +=1

    x = 1
    # if history['powerballs'] exits
    for i in history['powerballs']:
        tmp = i
        tmp.reverse()
        pb[str(x)] = tmp.index(1)
        x += 1

    return {'mainballs': mb, 'powerballs': pb}

def analyze():
    recency = timesSinceLastDraw()
    freq = json.load(open("Frequencies.json"))

    mb = {}
    pb = {}

    format = ['ball', 'prob', 'times since last drawn', 'expected difference between drawings', 'overdue?']

    x = 1
    for i in list(recency['mainballs'].keys()):
        r = recency['mainballs'][i]
        p = float(freq['mainballs'][i])/float(freq['numdraws'])
        # prob = geomPDF(p, r)
        prob = geomCDF(p, r)
        exp = geomEXPECTED(p)
        mb[str(x)] = [x, prob, r, exp, r > exp] # format [prob, times since last drawn, expected drawing, overdue?]
        x += 1

    x = 1
    for i in list(recency['powerballs'].keys()):
        r = recency['powerballs'][i]
        p = float(freq['powerballs'][i])/float(freq['numdraws'])
        # prob = geomPDF(p, r)
        prob = geomCDF(p, r)
        exp = geomEXPECTED(p)
        pb[str(x)] = [x, prob, r, exp, r > exp] # format [prob, times since last drawn, expected drawing, overdue?]
        x += 1

    writeCSV("mainball.csv", mb, format, True)
    writeCSV("powerball.csv", pb, format, True)


def writeCSV(filename, data, format, sorted=False):
    f = open(filename, "w")
    L = []
    L.append(str(format).replace("'", "")[1:-1] + '\n')

    vals = list(data.values())
    if sorted:
        vals.sort(key = lambda x: x[1])
        vals.reverse()

    for i in vals:
        L.append(str(i)[1:-1] + '\n')

    f.writelines(L)
    f.close()

# 0<=p<=1
def geomEXPECTED(p):
    return 1/p

# k >= 1 and 0<=p<=1
def geomPDF(p, k):
    return ((1-p)**k)*p

# k >= 1 and 0<=p<=1
def geomCDF(p, k):
    return 1 - (1-p)**(k+1)

if __name__ == '__main__':
    analyze()
    