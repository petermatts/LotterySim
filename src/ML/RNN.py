import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity='tanh', num_layers=1, balltype='mainball'):
        """
        @param input_size: int = expected number of features in RNN input
        @param hidden_size: int = output size of RNN and input size of linear layer
        @param output_size: int = number of balls
        @param nonlinearity: str = nonlinearity to use in RNN
        @param num_layers: int = number of RNN layers
        @param balltype: str = 'mainball' or 'specialball'
        """
        super(RNN, self).__init__()

        self.balltype = balltype
        # self.output_size = output_size

        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity=nonlinearity, num_layers=num_layers, device=device)
        self.lin = torch.nn.Linear(hidden_size, output_size, device=device)
        self.sm = torch.nn.Softmax()

        self.model = torch.nn.Sequential()

    def save(self, path: str):
        assert path.endswith('.pt')
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        assert path.endswith('.pt')
        torch.load(path)
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def predict(self, X: torch.Tensor):
        num = 5 if self.balltype == 'mainball' else 1
        balls = torch.arange(1, self.output_size+1, device=device)
        return torch.sort(balls[torch.topk(self.forward(X), num)]).values

    def loss(self):
        pass

    def learn(self):
        pass


if __name__ == '__main__':
    model = RNN(10, 26, 26)
    model.save()
