from pathlib import Path
from dataloader.dataset import GenomeDataset
from dataloader.dataloader import create_dataloader
from config import dataset as dt_cfg


def test_batch_data_size():
    data_path = Path('dataset/03.csv')
    batch_size = 16
    dt_cfg.DataLoader.batch_size = batch_size
    dataset = GenomeDataset(dt_cfg, data_path, resampling=True)
    dataloader = create_dataloader(dataset, dt_cfg.DataLoader)
    x, y = next(iter(dataloader))
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert len(y.shape) == 2

    assert x.shape == (batch_size, dt_cfg.DataSampling.n_sample, dataset.data_size)


def test_batch_data_size_no_resampling():
    data_path = Path('dataset/03.csv')
    batch_size = 16
    dt_cfg.DataLoader.batch_size = batch_size
    dataset = GenomeDataset(dt_cfg, data_path, resampling=False)
    dataloader = create_dataloader(dataset, dt_cfg.DataLoader)
    x, y = next(iter(dataloader))
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert len(y.shape) == 2

    assert x.shape == (batch_size, dataset.data_size)
