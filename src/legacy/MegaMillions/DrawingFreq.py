import json

def generateFreqDict():
    history = json.load(open("History.json"))
    
    mballs = {}
    Mballs = {}
    draws = len(history['megaballs'][0])

    for i in range(len(history['mainballs'])):
        freq = 0
        for j in history['mainballs'][i]:
            if j == 1:
                freq += 1
        mballs[str(i+1)] = freq

    for i in range(len(history['megaballs'])):
        freq = 0
        for j in history['megaballs'][i]:
            if j == 1:
                freq += 1
        Mballs[str(i+1)] = freq

    frequency = {
        "mainballs": mballs,
        "megaballs": Mballs,
        "numdraws": draws
    }

    with open("Frequencies.json", "w") as outfile:
        json.dump(frequency, outfile, indent=4)
        

if __name__ == "__main__":
    generateFreqDict()