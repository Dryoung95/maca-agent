import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.expm1(x)
        x = nn.Softmax(dim=1)(x)
        return x


batch_size = 64
width = 128
height = 1024


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    # Since there are no learnable parameters in the model, this function returns an empty list
    return []
