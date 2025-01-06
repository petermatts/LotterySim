import torch.nn as nn

# may need to get smarter about this?

class LotteryTransformer(nn.module):
    def __init__(self):
        # embed the numbers going in? 
        
        transformer = nn.Transformer()
        linear = nn.Linear()
        softmax = nn.Softmax()

        self.model = nn.Sequential(transformer, linear, softmax)

    def forward(self, x):
        return self.model(x)
