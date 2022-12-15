import typing


class DataSampling:
    replace: bool = True
    n_sample: int = 32
    random_state: int = None


class DataLoader:
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True
