from pathlib import Path
import pandas as pd
from dataloader.dataset import GenomeDataset
from dataloader.cleaning import clean_genome_df
from config import dataset as dt_cfg


def test_len_dataset():
    data_path = Path('dataset/03.csv')
    df = pd.read_csv(data_path, index_col=0, low_memory=False)
    x, y = clean_genome_df(df)
    dataset = GenomeDataset(dt_cfg, data_path, resampling=True)
    assert len(x) == len(dataset)


def test_data_shape():
    number_of_samples = 21
    dt_cfg.DataSampling.n_sample = number_of_samples
    data_path = Path('dataset/03.csv')
    dataset = GenomeDataset(dt_cfg, data_path, resampling=True)
    x, y = dataset[0]
    x1, y1 = dataset[22]

    assert x.shape == x1.shape == (number_of_samples, dataset.data_size)


def test_custom_features():
    from config import features
    number_of_samples = 45
    dt_cfg.DataSampling.n_sample = number_of_samples
    data_path = Path('dataset/03.csv')
    dataset = GenomeDataset(dt_cfg, data_path, features.MRMR_SELECTED_FEATURES_10, resampling=True)
    x, y = dataset[0]
    x1, y1 = dataset[22]

    assert x.shape == x1.shape == (number_of_samples, len(features.MRMR_SELECTED_FEATURES_10))


def test_with_no_resampling():
    from config import features
    data_path = Path('dataset/03.csv')
    dataset = GenomeDataset(dt_cfg, data_path, features.MRMR_SELECTED_FEATURES_10, resampling=False)
    x, y = dataset[0]
    x1, y1 = dataset[31]

    assert x.shape == x1.shape == (len(features.MRMR_SELECTED_FEATURES_10),)