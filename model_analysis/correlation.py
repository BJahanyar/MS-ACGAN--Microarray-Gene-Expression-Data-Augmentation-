from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

from config import train as train_cfg, features, model, dataset as dt_cfg
from dataloader import GenomeDataset
from models.generator import Generator
from utils.cls_utils import generate_data
from utils.train_utils import get_mean_for_each_label


def plot_heatmap(corr_matrix):
    # create heatmap using Seaborn
    sns.heatmap(np.round(corr_matrix, 2), annot=False, cmap='coolwarm', square=True)

    # customize the plot
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # show the plot
    plt.show()


def correlation_between_datasets(human_data: np.ndarray, generated_data: np.ndarray):
    corr_matrix = np.zeros((human_data.shape[1], generated_data.shape[1]))
    for i in range(human_data.shape[1]):
        for j in range(generated_data.shape[1]):
            corr_matrix[i, j] = np.corrcoef(human_data[:, i], generated_data[:, j])[0, 1]
    print("The Pearson correlation coefficient is:", )
    print(corr_matrix)
    return corr_matrix


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
    dataset = GenomeDataset(dt_cfg, data_path, selected_features)
    x = dataset.scaler.fit_transform(dataset.x)
    y = dataset.y

    label_means, label_stds, unique_labels = get_mean_for_each_label(x, y)
    train_data_len = len(x)
    generators = [
        Generator(
            in_features=train_cfg.LATENT_SIZE,
            out_features=len(selected_features),
            hidden_layers=model.G_HIDDEN_LAYERS,
        ) for _ in range(4)
    ]
    use_mean_std = [False, True, False, True]
    data_len = len(x)
    for gen_i, gen in enumerate(generators):
        gen.load_state_dict(torch.load(generator_paths[gen_i][1]))
        gen.eval()
        print(f'Generator: {generator_paths[gen_i][0]}')

        if use_mean_std[gen_i]:
            fake_train, fake_train_class = generate_data(gen, data_len, unique_labels, label_means, label_stds)
        else:
            fake_train, fake_train_class = generate_data(gen, data_len)

        print(f"Number of data: {data_len}")
        corr_matrix = correlation_between_datasets(x, fake_train)
        original_corr_matrix = correlation_between_datasets(x, x)
        fake_corr_matrix = correlation_between_datasets(fake_train, fake_train)
        plot_heatmap(corr_matrix)
        plot_heatmap(original_corr_matrix)
        plot_heatmap(fake_corr_matrix)
