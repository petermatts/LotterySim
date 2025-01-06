import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Data Generator. Uses number frequencies from lottery \
                                 data to generate (simulate) more drawings to increase data size")

lotto_type = parser.add_mutually_exclusive_group(required=True)
lotto_type.add_argument("--megamillions", action='store_true', help="Generate data for the MegaMillions lottery")
lotto_type.add_argument("--powerball", action='store_true', help="Generate data for the Powerball lottery")

parser.add_argument("--n", type=int, default=10000, help="The number of additional drawings to simulate")
parser.add_argument("--adaptive", action="store_true", help="Adjust the distibution of balls according to newly generated data")


args = vars(parser.parse_args())

DATAPATH = (Path(__file__).parent / "../../Data").resolve()
LOTTERY = ""

if args['megamillions']:
    DATAPATH = DATAPATH / "MegaMillions"
    Lottery = "MegaMillions"
elif args['powerball']:
    DATAPATH = DATAPATH / "Powerball"
    LOTTERY = "Powerball"
# add more lotteries as necessary


def generateData(datapath: Path, lottery: str, n: int = 10000, adaptive: bool = False):
    with open(datapath / "frequencies.json", "r") as file:
        freq = json.load(file)

    ndraws: int = freq["numdraws"]
    mainballs: dict = freq["mainballs"]
    specialballs: dict = freq["specialballs"]

    with open(datapath / "drawings.csv", "r") as file:
        data = file.read()

    # do the drawing
    mb = list(mainballs.keys())
    mbp = np.array(list(mainballs.values())) / float(ndraws)
    mbp /= np.sum(mbp)

    sb = list(specialballs.keys())
    sbp = np.array(list(specialballs.values())) / float(ndraws)

    for _ in range(n):
        mbs = np.sort(np.random.choice(mb, size=5, p=mbp, replace=False))
        for b in mbs:
            data += f"{b :>2},"
        
        sbs = np.random.choice(sb, p=sbp)
        data += f"{sbs :>2}\n"

        if adaptive:
            ndraws += 1
            specialballs[sbs] += 1
            for b in mbs:
                mainballs[b] += 1

            mb = list(mainballs.keys())
            mbp = np.array(list(mainballs.values())) / float(ndraws)
            mbp /= np.sum(mbp)

            sb = list(specialballs.keys())
            sbp = np.array(list(specialballs.values())) / float(ndraws)

    with open(datapath / f"simdata{'_adaptive' if adaptive else ''}.csv", "w") as file:
        file.write(data)


generateData(datapath=DATAPATH, lottery=LOTTERY, n=args['n'], adaptive=args['adaptive'])
