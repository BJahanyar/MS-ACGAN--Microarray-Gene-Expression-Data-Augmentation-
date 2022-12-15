from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_features: int, num_classes: int, hidden_layers: list):
        super(MLPClassifier, self).__init__()
        self._hidden_layers = hidden_layers
        layers = []

        for i, (in_feature, out_feature) in enumerate(
                zip([input_features] + self._hidden_layers, self._hidden_layers + [num_classes])):
            layers.append(nn.Linear(in_feature, out_feature))
            if i < len(self._hidden_layers):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.layers(input_tensor)

