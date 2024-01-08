from scipy.stats import geom
import Helper
import os
import json

class Geometric:
    def __init__(self):
        with open('Frequencies.json', 'r') as f:
            self.frequencies = json.load(f)
        
        with open('History.json', 'r') as f:
            self.history = json.load(f)

        self.timesSinceLastDraw = Helper.timesSinceLastDraw(self.history)
            
        print(self.timesSinceLastDraw)
        print(self.frequencies)


    def analze(self):
        common_keys = set(self.frequencies.keys()).intersection(set(self.timesSinceLastDraw.keys()))
        # print(common_keys)

        numdraws = self.frequencies['numdraws']

        probs = {}
        means = {}