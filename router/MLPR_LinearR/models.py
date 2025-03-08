import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(input_size, hidden_size))
        # self.layers.append(torch.nn.Dropout(p=0.5, inplace=False))
        self.layers.append(torch.nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(hidden_size, hidden_size))
        # self.layers.append(torch.nn.Dropout(p=0.5, inplace=False))
        self.layers.append(torch.nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            
        return x

    
if __name__ == '__main__':
    print()