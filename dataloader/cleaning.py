import typing

import numpy as np
import pandas as pd


def clean_genome_df(df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
    y = df.iloc[0].apply(str.lower)
    y_uniques = np.unique(y).tolist()
    y_int = np.zeros((len(y), 1), dtype=np.int32)
    for i, cls in enumerate(y_uniques):
        y_int[y == cls] = i

    x = df.iloc[1:].values
    x = x.astype(np.float32)
    x = x.T
    return x, y_int

