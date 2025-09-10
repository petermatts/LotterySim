from pandas import read_csv
from pathlib import Path
from torch import tensor, Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

class LotteryDataset(Dataset):
    def __init__(self, csv_path: Path, dims: tuple[int, int], lookback: int = 10):
        self.lookback = lookback

        self.data = read_csv(csv_path)
        self.dims = dims
        assert len(self.dims) == 2

    def __len__(self) -> int:
        return len(self.data) - self.lookback - 1

    def __getitem__(self, index: int) -> tuple[Tensor]:
        data = tensor(self.data.iloc[index:index+self.lookback, :6].to_numpy()).float()

        return (
            data, 
            (
                F.one_hot(data[:, :5][-1].long(), self.dims[0]).sum(dim=0).float(),
                data[-1,-1].long()
            )
        )


if __name__ == "__main__":
    example_path = Path(__file__) / "../../../data/Powerball/drawings.csv"
    ds = LotteryDataset(example_path.resolve(), dims=(70,27), lookback = 5)

    print(ds.data)
    print(len(ds.data))

    print(ds.__getitem__(0))


