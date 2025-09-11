from torch import nn, Tensor

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: tuple[int], num_layers: int = 1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.ModuleList([
            nn.Linear(hidden_size, out_size) for out_size in output_size
        ])

    def forward(self, x: Tensor) -> tuple[Tensor]:
        out, _ = self.rnn(x)  # (batch, seq_len, hidden)
        out = out[:, -1, :]   # last time step
        out = [fc(out) for fc in self.fc]
        return tuple(out)
    