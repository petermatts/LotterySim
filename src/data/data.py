import argparse
import os
import glob
from pathlib import Path

from functions import scrape, generateHistDict, generateFreqDict

DATAPATH = (Path(__file__).parent / "../../Data").resolve()

# make args
parser = argparse.ArgumentParser()
lotto_type = parser.add_mutually_exclusive_group(required=True)
lotto_type.add_argument("--megamillions", action='store_true', help="Gather data for the MegaMillions lottery")
lotto_type.add_argument("--powerball", action='store_true', help="Gather data for the Powerball lottery")

args = vars(parser.parse_args())

if args['megamillions']:
    from megamillions import WHITEBALLS, REDBALLS, URL, START
    DATAPATH = DATAPATH / "MegaMillions"
elif args['powerball']:
    from powerball import WHITEBALLS, REDBALLS, URL, START
    DATAPATH = DATAPATH / "Powerball"
# add additional lotteries here

print("Removing Old Data")
old_data = glob.glob(str(DATAPATH / "*"))
for f in old_data:
    os.remove(f)

print("Gathering Data:")
scrape(DATAPATH, URL, START)

print("\nGenerating History Dictionary")
generateHistDict(DATAPATH, WHITEBALLS, REDBALLS)

print("\nGenerating Frequency Dictionary")
generateFreqDict(DATAPATH, WHITEBALLS, REDBALLS)

print("\nDone.")