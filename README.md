# LotterySim

## Running PowerBall

### Scraping the Drawing Data

From the directory ```/src/Powerball``` run ```.\scrape.bat``` (Windows only for now, MacOS will be ```./scrape.command``` or ```./scrape.sh```) 

Or from that directory manually run ```python3 DrawingFreq.py``` and ```python3 DrawingHist.py```

### Running the Analysis

```python3 Analysis.py```

This will out put numbers to CSV files in accordance to their geometic probabilities based on when they were last drawn.

The files are called ```mainball.csv``` for the main balls and ```poewrball.csv``` for the powerballs.

## Running MegaMillions

MegaMillions code and analysis has not yet been implemented. Come back soon.