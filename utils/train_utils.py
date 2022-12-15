from pathlib import Path
import typing

import numpy as np
import torch

from config import train as train_cf


def compute_acc(predictions: torch.Tensor, labels: torch.Tensor):
    preds_ = torch.where(predictions > 0.5, 1, 0)
    correct = preds_.eq(labels.data).cpu().sum()
    acc = correct / len(labels.data) * 100.0
    return acc


def create_real_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.ones((batch_size, 1), dtype=torch.float32, device=device)


def create_fake_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch_size, 1), dtype=torch.float32, device=device)


def create_eval_data(batch_size: int, device: torch.device) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    eval_noise = torch.randn((batch_size, train_cf.LATENT_SIZE, 1, 1), dtype=torch.float32, device=device)
    eval_label = torch.randint(0, train_cf.NUM_CLASSES, (batch_size, 1), dtype=torch.float32, device=device)
    eval_noise = eval_noise + eval_label
    return eval_noise, eval_label


class Train:
    def __init__(
            self,
            generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            g_optim,
            d_optim,
            source_loss,
            class_loss,
            device: torch.device,
    ):
        self._generator = generator
        self._discriminator = discriminator
        self._g_optim = g_optim
        self._d_optim = d_optim
        self.source_loss = source_loss
        self.class_loss = class_loss
        self.device = device

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    def generator_zero_grad(self):
        for param in self._generator.parameters():
            param.grad = None

    def discriminator_zero_grad(self):
        for param in self._discriminator.parameters():
            param.grad = None

    def train_dis_with_real_data(self, data, labels, real_labels):
        _source, _class = self._discriminator(data)
        source_loss = self.source_loss(_source, real_labels)
        class_loss = self.class_loss(_class, labels.float())
        dis_real_loss = source_loss + class_loss
        dis_real_loss.backward()
        self._d_optim.step()
        return dis_real_loss, _source, _class

    def train_dis_with_fake_data(self, fake_labels, noise, fake_classes):
        fake_data = self._generator(noise)
        _source, _class = self._discriminator(fake_data.detach())
        source_loss = self.source_loss(_source, fake_labels)
        class_loss = self.class_loss(_class, fake_classes.float())
        dis_fake_loss = source_loss + class_loss
        dis_fake_loss.backward()
        self._d_optim.step()
        return dis_fake_loss, _source, _class, fake_data

    def train_generator(self, fake_data, real_labels, fake_classes):
        _source, _class = self._discriminator(fake_data)
        source_loss = self.source_loss(_source, real_labels)
        class_loss = self.class_loss(_class, fake_classes.float())
        gen_loss = source_loss + class_loss
        gen_loss.backward()
        self._g_optim.step()
        return gen_loss


def save_model(model: torch.nn.Module, path: Path):
    torch.save(model.state_dict(), path)


def get_mean_for_each_label(data: np.ndarray, labels: np.ndarray):
    unique_labels = np.unique(labels)
    label_means = []
    label_stds = []
    for l in unique_labels:
        mean = data[labels.ravel() == l].mean()
        std = data[labels.ravel() == l].std()
        label_means.append(mean)
        label_stds.append(std)
        print(f"Label ({l}) mean is: ({mean}), STD: ({std})")
    return label_means, label_stds, unique_labels.tolist()
