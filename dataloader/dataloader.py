from torch.utils.data import DataLoader, Dataset

__all__ = (
    'create_dataloader',
)


def create_dataloader(dataset: Dataset, options):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=options.batch_size,
        shuffle=options.shuffle,
        sampler=None,
        batch_sampler=None,
        num_workers=options.num_workers,
        collate_fn=None,
        pin_memory=options.pin_memory,
    )
    return dataloader
