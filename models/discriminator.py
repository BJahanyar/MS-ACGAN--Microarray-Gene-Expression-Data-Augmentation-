import torch
from pydantic import typing
from torch import nn


class ACDiscriminator(nn.Module):
    def __init__(self, num_classes: int, input_features: int, hidden_layers: list):
        super(ACDiscriminator, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers

        self.sigmoid = nn.Sigmoid()
        self.cls_sigmoid = nn.Sigmoid()

        self.fc_source = nn.Linear(self.hidden_layers[-1], 1)
        self.fc_class = nn.Linear(self.hidden_layers[-1], num_classes)

        layers = []

        for i in range(len(self.hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(in_features=input_features, out_features=self.hidden_layers[i]))
            else:
                layers.append(nn.Linear(in_features=self.hidden_layers[i - 1], out_features=self.hidden_layers[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        features = self.layers(data)
        classes = self.cls_sigmoid(self.fc_class(features))
        source = self.sigmoid(self.fc_source(features))
        return source, classes


class BasicDiscriminator(nn.Module):
    def __init__(self, input_features: int, hidden_layers: list):
        super(BasicDiscriminator, self).__init__()
        self.input_features = input_features
        self.hidden_layers = hidden_layers

        self.sigmoid = nn.Sigmoid()
        layers = []

        for i in range(len(self.hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(in_features=self.input_features, out_features=self.hidden_layers[i]))
            else:
                layers.append(nn.Linear(in_features=self.hidden_layers[i - 1], out_features=self.hidden_layers[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)
        self.fc_source = nn.Linear(self.hidden_layers[-1], 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc_source(self.layers(data)))
