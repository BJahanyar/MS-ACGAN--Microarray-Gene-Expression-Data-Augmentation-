import typing

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold

__all__ = (
    'open_csv_data',
    'clean_genome_data',
    'normalize_each_sample',
    'normalize_each_feature',
    'train_test_split',
    'feature_selection',
    'standard_scaler',
    'read_excel_data',
)


def read_excel_data(file: str):
    return pd.read_excel(file, index_col=0)


def open_csv_data(file: str):
    return pd.read_csv(file, index_col=0)


def clean_genome_data(df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
    y = df.iloc[0].apply(str.lower)
    x = df.iloc[1:].values
    x_float = x.astype(np.float32)
    x_transpose = x_float.T
    y_uniques = np.unique(y).tolist()
    y_int = np.zeros((len(y), 1), dtype=np.uint8)
    for i, cls in enumerate(y_uniques):
        y_int[y == cls] = i

    return x_transpose, y_int


def normalize_each_sample(data: np.ndarray) -> np.ndarray:
    return normalize(data, axis=1)


def normalize_each_feature(data: np.ndarray) -> np.ndarray:
    return normalize(data, axis=0)


def standard_scaler(data: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    return model_selection.train_test_split(x, y, test_size=test_size, shuffle=True)


def feature_selection(data: np.ndarray, variance: float) -> np.ndarray:
    feature_selector = VarianceThreshold(threshold=variance)
    return feature_selector.fit_transform(data)
