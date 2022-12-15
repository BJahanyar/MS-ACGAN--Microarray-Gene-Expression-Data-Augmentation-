import torch

from models import (Generator, MLPClassifier, ACDiscriminator, BasicDiscriminator, CNN1Classifier)


def test_generator():
    in_features = 8
    out_features = 512
    hidden_layers = [32, 64, 128]
    gen = Generator(
        in_features=in_features,
        out_features=out_features,
        hidden_layers=hidden_layers,
    )
    random_input = torch.randn(1, in_features)
    gen_output: torch.Tensor = gen(random_input)
    assert gen_output.shape == (1, 512)
    assert gen_output.requires_grad
    print(gen)


def test_classifier():
    in_features = 28
    num_classes = 2
    batch_size = 12
    hidden_layers = [32, 64]
    cls = MLPClassifier(in_features, num_classes, hidden_layers)
    random_input = torch.randn(batch_size, in_features)
    predict = cls(random_input)
    assert predict.shape == (batch_size, num_classes)
    assert 0 <= predict.max() <= 1


def test_acdiscriminator():
    num_classes = 11
    batch = 5
    input_features = 512
    hidden_layers = [256, 128, 32]

    dis = ACDiscriminator(num_classes, input_features, hidden_layers)
    random_data = torch.randn(batch, input_features)
    pred_source, pred_classes = dis(random_data)
    assert pred_source.shape == (batch, 1)
    assert pred_classes.shape == (batch, num_classes)
    assert 0 <= torch.min(pred_source) and torch.max(pred_source) <= 1
    assert 0 <= torch.min(pred_classes) and torch.max(pred_classes) <= 1
    assert pred_classes.requires_grad
    assert pred_source.requires_grad


def test_basic_discriminator():
    batch = 5
    input_features = 512
    hidden_layers = [256, 128, 32]
    dis = BasicDiscriminator(
        input_features,
        hidden_layers,
    )
    random_data = torch.randn(batch, input_features)
    pred_source = dis(random_data)
    assert pred_source.shape == (batch, 1)
    assert 0 <= torch.min(pred_source) and torch.max(pred_source) <= 1
    assert pred_source.requires_grad


def test_save_model():
    in_features = 8
    out_features = 512
    hidden_layers = [32, 64, 128]
    gen = Generator(
        in_features=in_features,
        out_features=out_features,
        hidden_layers=hidden_layers,
    )


def test_cnn_classifier():
    in_channels = 3
    in_features = 64
    num_classes = 2
    batch_size = 12
    hidden_layers = [32, ]
    cls = CNN1Classifier(in_channels, hidden_layers, num_classes)
    random_input = torch.randn(batch_size, in_channels, in_features)
    predict = cls(random_input)
    assert predict.shape == (batch_size, num_classes)
    assert 0 <= predict.max() <= 1
