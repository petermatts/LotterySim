from torch import (
    nn,
    Tensor,
    device,
    cuda,
    sort,
    topk
)

device = device("cuda" if cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: tuple[int], num_layers: int = 1):
        """
        """

        assert len(output_size) == 2

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # One classifier per ball
        self.fc = nn.ModuleList([
            nn.Linear(hidden_size, out_size) for out_size in output_size
        ])


    def forward(self, x: Tensor) -> tuple[Tensor]:
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        last_out = out[:, -1, :]  # take last time step

        out = [fc(out) for fc in self.fc]

        return tuple(out)

    # def predict(self, X: Tensor):
    #     num = 5 if self.balltype == 'mainball' else 1
    #     balls = torch.arange(1, self.output_size+1, device=device)
    #     return torch.sort(balls[torch.topk(self.forward(X), num)]).values
