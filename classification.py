import os

from config import model
from utils.cls_utils import (logistic_regression_classifier, nn_classifier, cnn1d_classifier,
                             knn_classifier, svm_classifier, train_classifiers_with_fake_data)

os.makedirs(model.SAVE_PATH, exist_ok=True)

if __name__ == '__main__':
    # logistic_regression_classifier()
    # nn_classifier(calibrate=True)
    # cnn1d_classifier(calibrate=True)
    # knn_classifier(calibrate=True)
    svm_classifier(calibrate=False)
    # grid_search_nn_classifier()
    # train_concatenate_datasets()
    # train_classifiers_with_fake_data()
