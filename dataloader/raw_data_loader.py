import typing
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = (
    'GenomeLoader',
    'AlzheimerLoader',
    'raw_dataloader_generator',
    'GenomeCancerLoader',
)


class BaseRawDataLoader(ABC):

    @abstractmethod
    def load_data(self, path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        ...

    def preprocess_data(self, path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        x, y = self.load_data(path)
        return x,  y


class GenomeLoader(BaseRawDataLoader):
    def __init__(self):
        super(GenomeLoader, self).__init__()

    def load_data(self, path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        if path.endswith('csv'):
            df = pd.read_csv(path, index_col=0)
        elif path.endswith('xlsx'):
            df = pd.read_excel(path, index_col=0)
        else:
            raise TypeError('File type not supported')
        return self.clean_data(df)

    @staticmethod
    def clean_data(df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
        y = df.iloc[0].apply(str.lower)
        y_uniques = np.unique(y).tolist()
        y_int = np.zeros((len(y), 1), dtype=np.uint32)
        for i, cls in enumerate(y_uniques):
            y_int[y == cls] = i

        x = df.iloc[1:].values
        x = x.astype(np.float32)
        x = x.T
        return x, y_int


class AlzheimerLoader(BaseRawDataLoader):
    def __init__(self):
        super(AlzheimerLoader, self).__init__()

    def load_data(self, path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        if path.endswith('csv'):
            df = pd.read_csv(path, index_col=0)
        elif path.endswith('xlsx'):
            df = pd.read_excel(path, index_col=0)
        else:
            raise TypeError('File type not supported')
        return self.clean_data(df)

    @staticmethod
    def clean_data(df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
        y = df.iloc[0].apply(str.lower)
        y_int = np.zeros((len(y), 1), dtype=np.uint32)
        for i, cls in enumerate(y):
            if 'control' in cls:
                y_int[i] = 0
            elif 'case' in cls:
                y_int[i] = 1
            elif 'neuro' in cls:
                y_int[i] = 2

        x = df.iloc[1:].values
        x = x.astype(np.float32)
        x = x.T
        return x, y_int


class GenomeCancerLoader(GenomeLoader):

    @staticmethod
    def clean_data(df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
        y = df.iloc[0].apply(str.lower)
        y_int = np.zeros((len(y), 1), dtype=np.uint32)
        for i, cls in enumerate(y):
            if 'control' in cls:
                y_int[i] = 0
            elif 'case' in cls:
                y_int[i] = 1
            elif 'neuro' in cls:
                y_int[i] = 2

        df.dropna(axis=0, inplace=True)
        x = df.iloc[1:].values
        x = x.astype(np.float32)
        x = x.T
        return x, y_int


def raw_dataloader_generator(creator: BaseRawDataLoader, path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    return creator.preprocess_data(path)


if __name__ == '__main__':
    x, y = raw_dataloader_generator(GenomeCancerLoader(), '../dataset/GSE11223_series_matrix.xlsx')
