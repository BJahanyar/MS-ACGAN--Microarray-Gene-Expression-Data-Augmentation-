import os
from pathlib import Path
from utils.train_utils import save_model
import torch


def test_save_model():
    save_path = Path('test_saved_model', 'test_model.pth')
    save_path.parent.mkdir(exist_ok=True)
    test_model = torch.nn.Conv2d(32, 32, (3, 3), (1, 1))
    save_model(test_model, save_path)
    assert save_path.exists()
    os.remove(save_path)
    assert not save_path.exists()
    save_path.parent.rmdir()
    assert not save_path.parent.exists()

