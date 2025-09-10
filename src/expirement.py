import argparse
from pathlib import Path
import torch.nn as nn
from data import LotteryDataset, getDataSplit
from ML import LSTM, Trainer
from resources import getOptimizer

# todo add argparse
# todo add expirement config (yaml, json, or toml)

# todo write a predict next draw function given a model

if __name__ == "__main__":
    ds_path = (Path(__file__).parent / "../data/Powerball/drawings.csv").resolve()
    
    from data.powerball import WHITEBALLS, REDBALLS
    dims = (WHITEBALLS+1, REDBALLS+1)

    ds = LotteryDataset(ds_path, dims=dims, lookback=10)
    model = LSTM(6, 100, dims)

    train, val, test = getDataSplit(ds, batch_size=16)

    # for (inputs, targets) in train:
    #     print(inputs.shape)
    #     print(targets[0].shape, targets[1].shape)
    #     print()

    machine = Trainer(
        model,
        train_loader = train,
        val_loader = val,
        optimizer = getOptimizer("Adam", model),
        num_epochs = 1,
        loss_fn=[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()]
    )

    machine.learn()
    # machine._validate()
