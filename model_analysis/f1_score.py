from sklearn.metrics import f1_score
import numpy as np
from sklearn.utils import resample


def bootstrapping_f1_score(clf, x_test, y_test, bootstrap_number):
    bootstrap_f1_scores = []
    for _ in range(bootstrap_number):
        x_boot, y_boot = resample(x_test, y_test)
        y_predict = clf.predict(x_test)
        bootstrap_f1_scores.append(f1_score(y_boot, y_predict))

    brier_mean = np.mean(bootstrap_f1_scores)
    brier_std = np.std(bootstrap_f1_scores)

    alpha = 0.95  # set confidence level
    lower_bound = brier_mean - 1.96 * brier_std / np.sqrt(bootstrap_number)
    upper_bound = brier_mean + 1.96 * brier_std / np.sqrt(bootstrap_number)
    middle_value = (upper_bound + lower_bound) / 2
    radius = (upper_bound - lower_bound) / 2

    # print confidence interval
    print(f"Confidence interval *(F1-Score)*: (alpha={alpha:.2f}): ({lower_bound:.4f}, {upper_bound:.4f})"
          f" | {middle_value:.4f} \u00B1 {radius:.4f}")


