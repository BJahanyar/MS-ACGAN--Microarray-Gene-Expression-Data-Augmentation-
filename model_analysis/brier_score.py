from pathlib import Path
import pickle

from sklearn.metrics import brier_score_loss
import numpy as np
from sklearn.utils import resample

from config import dataset as dt_cfg
from config import features
from dataloader import GenomeDataset


def calculate_brier_score(y_true, y_prob):
    """
    Calculates the Brier score for a binary classification problem using scikit-learn.

    Parameters:
    y_true (array-like): true binary labels (0 or 1)
    y_prob (array-like): predicted probabilities of the positive class (1)

    Returns:
    brier_score (float): the Brier score
    """
    brier_score = brier_score_loss(y_true, y_prob)
    return brier_score


def bootstrapping_brier_score(clf, x_test, y_test, bootstrap_number: int):
    bootstrap_brier_scores = []
    for _ in range(bootstrap_number):
        x_boot, y_boot = resample(x_test, y_test)
        y_proba = clf.predict_proba(x_boot)[:, 1]
        bootstrap_brier_scores.append(calculate_brier_score(y_boot, y_proba))

    brier_mean = np.mean(bootstrap_brier_scores)
    brier_std = np.std(bootstrap_brier_scores)

    alpha = 0.95  # set confidence level
    lower_bound = brier_mean - 1.96 * brier_std / np.sqrt(bootstrap_number)
    upper_bound = brier_mean + 1.96 * brier_std / np.sqrt(bootstrap_number)
    middle_value = (upper_bound + lower_bound) / 2
    radius = (upper_bound - lower_bound) / 2

    # print confidence interval
    print(f"Confidence interval *(Brier Score)*: (alpha={alpha:.2f}): ({lower_bound:.4f}, {upper_bound:.4f})"
          f" | {middle_value:.4f} \u00B1 {radius:.4f}")


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
        print(validation_model_path)
        bootstrapping_brier_score(
            val_cls,
            x,
            y,
            400,
        )
    #
    #     # predict probabilities for positive class
    #     y_predict = val_cls.predict_proba(x)[:, 1]
    #
    #     # Calculate Brier score
    #     brier_score = calculate_brier_score(y, y_predict)
    #     print(f"Model: {validation_model_path} --> Brier Score = {brier_score:.4f}")
