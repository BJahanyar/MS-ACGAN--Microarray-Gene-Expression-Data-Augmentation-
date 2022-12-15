import typing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from .cleaning import clean_genome_df


def open_file(data_path):
    if data_path.suffix == '.xlsx':
        return pd.read_excel(str(data_path), engine='openpyxl', index_col=0)
    elif data_path.suffix == '.csv':
        return pd.read_csv(str(data_path), index_col=0, low_memory=False)


class GenomeDataset(Dataset):
    def __init__(self, config, data_path: Path, sample_features: list = None, resampling: bool = False):
        super(GenomeDataset, self).__init__()
        self.config = config
        self.df = open_file(data_path)
        self.sample_feature = sample_features
        self.resampling = resampling

        if self.sample_feature is not None:
            self.x, self.y = clean_genome_df(self.df.loc[['!Sample_characteristics_ch1'] + self.sample_feature])
        else:
            self.x, self.y = clean_genome_df(self.df)

        self.scaler = StandardScaler()
        self._number_of_classes = 2
        self._data_size = self.x.shape[1]

    @property
    def data_size(self):
        return self._data_size

    @property
    def number_of_classes(self):
        return self._number_of_classes

    def __getitem__(self, index) -> typing.Tuple[np.ndarray, np.ndarray]:
        y = self.y[index]
        if self.resampling:
            x = resample(
                self.x[(self.y == y).squeeze(axis=1)],
                replace=self.config.DataSampling.replace,
                n_samples=self.config.DataSampling.n_sample,
                random_state=self.config.DataSampling.random_state,
            )
        else:
            x = self.x[index]
        return x, y

    def __len__(self):
        return len(self.x)
