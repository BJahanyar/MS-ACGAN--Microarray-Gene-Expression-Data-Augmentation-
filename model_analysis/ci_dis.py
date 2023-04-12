from pathlib import Path

import torch

from config import train as train_cfg, features, model, dataset as dt_cfg
from models.generator import Generator
from utils.cls_utils import generate_data
from utils.train_utils import get_mean_for_each_label
from dataloader import GenomeDataset
from .confidence_interval import ci_2_random_distributions


def calculate_ci(generator_paths: list, data_path, _features):
    dataset = GenomeDataset(dt_cfg, data_path, _features)
    x = dataset.scaler.fit_transform(dataset.x)
    y = dataset.y

    label_means, label_stds, unique_labels = get_mean_for_each_label(x, y)
    train_data_len = len(x)
    generators = [
        Generator(
            in_features=train_cfg.LATENT_SIZE,
            out_features=len(_features),
            hidden_layers=model.G_HIDDEN_LAYERS,
        ) for _ in range(4)
    ]
    use_mean_std = [False, True, False, True]
    number_of_data = [train_data_len * i for i in (2, 5, 20, 50, 100)]
    for gen_i, gen in enumerate(generators):
        gen.load_state_dict(torch.load(generator_paths[gen_i][1]))
        gen.eval()
        print(f'Generator: {generator_paths[gen_i][0]}')

        for data_len in number_of_data:
            if use_mean_std[gen_i]:
                fake_train, fake_train_class = generate_data(gen, data_len, unique_labels, label_means, label_stds)
            else:
                fake_train, fake_train_class = generate_data(gen, data_len)

            print(f"Number of data: {data_len}")
            ci_2_random_distributions(fake_train, x)


if __name__ == '__main__':
    data_path = Path('dataset/8_202.csv')
    selected_features = features.SELECTED_FEATURES_64_8_202
    model_name_prefix = '_Generated'
    generator_paths = [
        ('G_no_mean_std_8_202', Path("SavedModels/2022-12-02 18_21_53acgan_without_mean_std_8_202.pth")),
        ('G_mean_std_8_202', Path("SavedModels/2022-12-02 18_25_32acgan_mean_std_8_202.pth")),
        ('Basic_G_no_mean_std_8_202', Path("SavedModels/2022-12-02 18_31_38basic_gan_without_mean_std_8_202.pth")),
        ('Basic_G_mean_std_8_202', Path("SavedModels/2022-12-02 18_33_10basic_gan_with_mean_std_8_202.pth")),
    ]
    calculate_ci(generator_paths, data_path, selected_features)
