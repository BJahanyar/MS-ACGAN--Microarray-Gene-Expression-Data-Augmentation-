from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def plot_loss_acc(results: np.ndarray, labels: list, save_path: Path = None):
    plt.figure(figsize=(20, 20))
    for i, label in enumerate(labels):
        plt.plot(results[i], label=label)
    plt.legend()
    # plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def test_plot_loss_acc():
    import numpy as np
    result = np.random.randn(3, 1000)
    labels = ['1', '2', '3']
    plot_loss_acc(result, labels)
