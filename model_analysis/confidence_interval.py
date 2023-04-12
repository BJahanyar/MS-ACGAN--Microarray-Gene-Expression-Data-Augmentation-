import os
import pickle
from pathlib import Path
import warnings

import numpy as np
from sklearn.utils import resample

from config import dataset as dt_cfg
from config import features
from dataloader import GenomeDataset

warnings.filterwarnings('ignore')


def ci_2_random_distributions(dis1, dis2):
    std1 = np.std(dis1)
    std2 = np.std(dis2)
    n_1 = len(dis1)
    n_2 = len(dis2)
    mean_1 = np.mean(dis1)
    mean_2 = np.mean(dis2)
    alpha = 0.95
    z = 1.96

    S_p = np.sqrt(((std1 ** 2) / n_1) + ((std2 ** 2) / n_2))
    lower_bound = (mean_1 - mean_2) - (z * S_p)
    upper_bound = (mean_1 - mean_2) + (z * S_p)

    middle_value = (upper_bound + lower_bound) / 2
    radius = (upper_bound - lower_bound) / 2

    # print confidence interval
    print(
        f"CI for two random variable (alpha={alpha:.2f}): ({lower_bound:.4f}, {upper_bound:.4f}) | {middle_value:.4f} \u00B1 {radius:.4f}")


def calculate_confidence_interval(
        clf,
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_bootstraps: int,
):
    bootstrap_scores = []

    for i in range(n_bootstraps):
        x_boot, y_boot = resample(x_test, y_test)
        y_pred = clf.predict(x_boot)
        accuracy = np.mean(y_pred == y_boot)
        bootstrap_scores.append(accuracy)

    mean_score = np.mean(bootstrap_scores)
    std_score = np.std(bootstrap_scores)

    alpha = 0.95  # set confidence level
    lower_bound = mean_score - 1.96 * std_score / np.sqrt(n_bootstraps)
    upper_bound = mean_score + 1.96 * std_score / np.sqrt(n_bootstraps)
    middle_value = (upper_bound + lower_bound) / 2
    radius = (upper_bound - lower_bound) / 2

    # print confidence interval
    print(
        f"Confidence interval *(ACC)* (alpha={alpha:.2f}): ({lower_bound:.4f}, {upper_bound:.4f}) | {middle_value:.4f} \u00B1 {radius:.4f}")


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
        print(f'******* Model: {os.path.basename(validation_model_path)} *******')
        calculate_confidence_interval(
            val_cls,
            x,
            y,
            400,
        )
