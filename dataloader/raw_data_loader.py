import typing
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = (
    'GenomeLoader',
    'raw_dataloader_generator',
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



def raw_dataloader_generator(creator: BaseRawDataLoader, path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    return creator.preprocess_data(path)
