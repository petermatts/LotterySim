import argparse

# make args
parser = argparse.ArgumentParser()
lotto_type = parser.add_mutually_exclusive_group(required=True)
lotto_type.add_argument("--megamillions", action='store_true', help="Gather data for the MegaMillions lottery")
lotto_type.add_argument("--powerball", action='store_true', help="Gather data for the Powerball lottery")

args = vars(parser.parse_args())

if args['megamillions']:
    from megamillions import scrape, generateHistDict, generateFreqDict
# elif args['powerball']:
#     from powerball import scrape, generateFreqDict
# add additional lotteries here

print("Gathering Data:")
scrape()

print("\nGenerating History Dictionary")
generateHistDict()

print("\nGenerating Frequency Dictionary")
generateFreqDict()

print("\nDone.")