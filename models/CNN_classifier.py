import torch
from torch import nn


class CNN1Classifier(nn.Module):
    def __init__(self, input_channel: int, hidden_layers: list, num_classes: int):
        super(CNN1Classifier, self).__init__()

        self.input_channel = input_channel
        self.num_classes = num_classes
        self._hidden_layers = hidden_layers

        layers = []

        for i, (in_channel, out_channel) in enumerate(
                zip([self.input_channel] + self._hidden_layers, self._hidden_layers + [self.num_classes])
        ):
            layers.append(nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                                    kernel_size=5, stride=5, ))

            if i < len(self._hidden_layers):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor):
        if len(input_tensor.shape) == 2:
            input_tensor.unsqueeze_(1)
        return self.layers(input_tensor).mean(dim=2)
