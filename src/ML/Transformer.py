import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity='tanh', num_layers=1, balltype='mainball'):
        """
        """

        self.balltype = balltype
        self.output_size = output_size

        self.lin = torch.nn.Linear(hidden_size, output_size, device=device)
        self.sm = torch.nn.Softmax()

    def forward(self, X: torch.Tensor):
        pass

    def predict(self, X: torch.Tensor):
        num = 5 if self.balltype == 'mainball' else 1
        balls = torch.arange(1, self.output_size+1, device=device)
        return torch.sort(balls[torch.topk(self.forward(X), num)]).values

    def loss(self):
        pass

    def learn(self):
        pass