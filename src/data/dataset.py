from pandas import read_csv
from pathlib import Path
from torch import tensor, Tensor
from torch.utils.data import Dataset

class LotteryDataset(Dataset):
    def __init__(self, csv_path: Path, lookback: int = 10):
        self.lookback = lookback

        self.data = read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.data) - self.lookback - 1

    def __getitem__(self, index: int) -> Tensor:
        return tensor(self.data.iloc[index:index+self.lookback, :6].to_numpy())


if __name__ == "__main__":
    example_path = Path(__file__) / "../../../data/Powerball/drawings.csv"
    ds = LotteryDataset(example_path.resolve(), lookback = 5)

    print(ds.data)
    print(len(ds.data))

    print(ds.__getitem__(0))


