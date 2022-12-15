import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: list):
        super(Generator, self).__init__()
        self._hidden_layers = hidden_layers
        layers = []

        for i, (in_feature, out_feature) in enumerate(
                zip([in_features] + self._hidden_layers, self._hidden_layers + [out_features])):
            layers.append(nn.Linear(in_feature, out_feature))
            if i < len(self._hidden_layers):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, z) -> torch.Tensor:
        return self.layers(z)
        