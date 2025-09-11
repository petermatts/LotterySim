from torch import nn, Tensor
from PositionalEncoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: tuple[int], num_layers: int = 1, nhead: int = 4, max_len: int = 5000):
        super(TransformerEncoder, self).__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.ModuleList([
            nn.Linear(hidden_size, out_size) for out_size in output_size
        ])

    def forward(self, x: Tensor) -> tuple[Tensor]:
        x = self.input_proj(x)               # (batch, seq_len, hidden)
        x = self.pos_encoder(x)
        out = self.encoder(x)                # (batch, seq_len, hidden)
        out = out[:, -1, :]                  # final timestep
        out = [fc(out) for fc in self.fc]
        return tuple(out)
    