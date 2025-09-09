from torch.utils.data import Subset, Dataset, DataLoader
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path


def getDataSplit(dataset: Dataset, split: list[float] = [0.7, 0.15, 0.15], batch_size: int = 64, ret_dataset: bool = False) -> tuple[DataLoader]:
    load_dotenv(Path(__file__).parent / "../../.env")
    seed = os.getenv("SEED")
    if seed is not None:
        np.random.seed(int(seed))
    else:
        print(f"No SEED environment variable found. Using random seed")

    assert len(split) == 3
    assert np.sum(split) == 1

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    train_idx, val_idx, test_idx = np.split(
        indices, [int(split[0]*len(indices)), int((split[0]+split[1])*len(indices))])

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    if ret_dataset:
        return train_dataset, val_dataset, test_dataset
    else:
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )
