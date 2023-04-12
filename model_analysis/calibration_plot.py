import pickle
from pathlib import Path
import os

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from config import dataset as dt_cfg
from config import features
from dataloader import GenomeDataset


def plot_calibration_curve(y_true, y_prob, name, n_bins=10, figsize=(8, 8), show=True):
    """
    Plots a calibration curve for a binary classification model using scikit-learn and matplotlib.

    Parameters:
    y_true (array-like): true binary labels (0 or 1)
    y_prob (array-like): predicted probabilities of the positive class (1)
    n_bins (int): number of bins to use in the calibration plot (default: 10)
    figsize (tuple): figure size in inches (default: (8, 8))

    Returns:
    None
    """
    # calculate calibration curve using scikit-learn
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # plot calibration curve using matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model Calibration')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    ax.set(xlabel='Mean Predicted Value', ylabel='Fraction of Positives', ylim=[-0.05, 1.05])
    ax.legend()
    ax.set_title(name)
    if show:
        plt.show()
    fig.savefig(os.path.join('calibration_figs', name + '.jpg'))
    # plt.close(fig)


def multi_plot_calibration_curve(y_true, y_prob, name, n_bins=10):
    """
    Plots a calibration curve for a binary classification model using scikit-learn and matplotlib.

    Parameters:
    y_true (array-like): true binary labels (0 or 1)
    y_prob (array-like): predicted probabilities of the positive class (1)
    n_bins (int): number of bins to use in the calibration plot (default: 10)
    figsize (tuple): figure size in inches (default: (8, 8))

    Returns:
    None
    """
    # calculate calibration curve using scikit-learn
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # plot calibration curve using matplotlib
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=name)


if __name__ == '__main__':

    test_data_path = Path('dataset/8_202_test.csv')
    train_data_path = Path('dataset/8_202_train.csv')
    validation_model_paths = [
        Path('SavedModels/1DCNN_8_202_calib.pickle'),
        Path('SavedModels/KNeighborsClassifier_8_202.pickle'),
        Path('SavedModels/LogisticRegression_8_202_calib.pickle'),
        Path('SavedModels/NeuralNetClassifier_8_202_calib.pickle'),
        Path('SavedModels/SVC_8_202_calib.pickle'),
    ]
    val_dataset = GenomeDataset(dt_cfg, test_data_path, features.SELECTED_FEATURES_64_8_202)
    x = val_dataset.scaler.fit_transform(val_dataset.x)
    y = val_dataset.y.ravel()

    for validation_model_path in validation_model_paths:
        '''1st load validation model'''
        val_cls = pickle.load(open(validation_model_path, 'rb'))

        # predict probabilities for positive class
        y_predict = val_cls.predict_proba(x)[:, 1]

        # Plot calibration curve
        # plot_calibration_curve(y, y_predict, os.path.basename(validation_model_path))
        multi_plot_calibration_curve(y, y_predict, os.path.basename(validation_model_path))

    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.legend()
    plt.title('Calibration Plots')
    plt.show()
