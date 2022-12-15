from datetime import datetime
from pathlib import Path

import numpy as np

from config import dataset as dt_cfg
from dataloader import GenomeDataset


def current_time() -> str:
    return str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))


def load_train_test_data(train_path: Path, test_path: Path, selected_features: list = None):
    train_dataset = GenomeDataset(dt_cfg, train_path, selected_features)
    x_train = train_dataset.scaler.fit_transform(train_dataset.x)
    y = train_dataset.y
    y_train = y.astype(np.int64).ravel()

    test_dataset = GenomeDataset(dt_cfg, test_path, selected_features)
    x_test = test_dataset.scaler.fit_transform(test_dataset.x)
    y = test_dataset.y
    y_test = y.astype(np.int64).ravel()
    return x_train, y_train, x_test, y_test
