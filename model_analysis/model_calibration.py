import os.path
import pickle
from pathlib import Path

import ml_insights as mli
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skorch import NeuralNetClassifier

from config import train as train_cfg, features, model, dataset as dt_cfg
from dataloader import GenomeDataset
from .confidence_interval import calculate_confidence_interval
from .f1_score import bootstrapping_f1_score
from models import MLPClassifier, CNN1Classifier
from models.generator import Generator
from utils.cls_utils import generate_data
from utils.train_utils import get_mean_for_each_label
from utils.utils import load_train_test_data
from .calibration_plot import plot_calibration_curve
from .brier_score import calculate_brier_score, bootstrapping_brier_score


def make_classification_algorithms(number_of_features, name_prefix: str = ''):
    classifiers = [
        ('LogisticRegression' + name_prefix, LogisticRegression(max_iter=1000, penalty='elasticnet', solver='saga', l1_ratio=0.1)),
        ('NeuralNetClassifier' + name_prefix, NeuralNetClassifier(
            MLPClassifier,
            max_epochs=100,
            lr=0.015,
            iterator_train__shuffle=True,
            module__input_features=number_of_features,
            module__num_classes=2,
            module__hidden_layers=[32],
            verbose=0,
        )),
        ('1DCNN' + name_prefix, NeuralNetClassifier(
            CNN1Classifier,
            max_epochs=100,
            lr=0.01,
            iterator_train__shuffle=True,
            module__input_channel=1,
            module__num_classes=2,
            module__hidden_layers=[24],
            verbose=0,
        )),
        ('KNeighborsClassifier' + name_prefix, KNeighborsClassifier(weights='uniform', n_neighbors=5, algorithm='kd_tree', leaf_size=50, p=1)),
        ('SVC' + name_prefix, SVC(probability=True, C=0.5, kernel='sigmoid', gamma='scale')),
    ]
    return classifiers


class ModelCalibration:
    def __init__(
            self,
            model,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            is_trained: bool = False,
            name: str = '',
    ):
        self._model = model
        self._calibrate_model = None
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._platt_result_sklearn = None
        self._platt_result_manual = None
        self._uncalib_result = None
        self.name = name

        self._is_trained = is_trained

    @property
    def x_train(self):
        return self._x_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def platt_result_sklearn(self) -> np.ndarray:
        if self._platt_result_sklearn is not None:
            return self._platt_result_sklearn
        raise ValueError('Not Calculated')

    @property
    def platt_result_manual(self) -> np.ndarray:
        if self._platt_result_manual is not None:
            return self._platt_result_manual
        raise ValueError('Not Calculated')

    @property
    def uncalibrate_result(self) -> np.ndarray:
        if self._uncalib_result is not None:
            return self._uncalib_result
        raise ValueError('Not Calculated')

    @property
    def model(self):
        return self._model

    @property
    def calibrate_model(self):
        return self._calibrate_model

    def fit(self):
        if not self._is_trained:
            self._model.fit(self._x_train, self._y_train)
        self.uncalibrate_proba()
        self.platt_scaling_sklearn()
        # self.platt_scaling_manual()

    def uncalibrate_proba(self):
        self._uncalib_result = self._model.predict_proba(self._x_test)[:, 1]

    def platt_scaling_sklearn(self) -> np.ndarray:
        self._calibrate_model = CalibratedClassifierCV(self._model, method='sigmoid', cv=5)
        self._calibrate_model.fit(self._x_train, self._y_train)
        y_prob_svm = self._calibrate_model.predict_proba(self._x_test)[:, 1]
        self._platt_result_sklearn = y_prob_svm

    def platt_scaling_manual(self):
        x_train = convert_array_to_df(self._x_train)

        cv_preds_train = mli.cv_predictions(self._model, x_train, self._y_train, clone_model=True, num_cv_folds=5)
        cv_preds_train1 = cv_preds_train[:, 1]
        testset_preds_uncalib_1 = self._model.predict_proba(self._x_test)[:, 1]

        lr_cv = LogisticRegression(C=99999999999, solver='lbfgs')
        lr_cv.fit(cv_preds_train1.reshape(-1, 1), self._y_train)

        testset_platt_probs_cv = lr_cv.predict_proba(testset_preds_uncalib_1.reshape(-1, 1))[:, 1]
        self._platt_result_manual = testset_platt_probs_cv

    def save_models(self):
        model_path = os.path.join(model.SAVE_PATH, self.name + '.pickle')
        calib_path = os.path.join(model.SAVE_PATH, self.name + "_calib.pickle")
        pickle.dump(
            self._model,
            open(model_path, 'wb')
        )
        pickle.dump(
            self._calibrate_model,
            open(calib_path, 'wb')
        )
        print(f'Model saved: {model_path}')


def convert_array_to_df(data: np.ndarray):
    assert len(data.shape) == 2
    number_of_columns = data.shape[1]
    return pd.DataFrame(data, columns=['A_' + str(i) for i in range(number_of_columns)])


def train_with_generated_data(generator_paths: list, classifiers: list, train_path, test_path, _features):
    x_train, y_train, x_test, y_test = load_train_test_data(train_path, test_path, _features)

    label_means, label_stds, unique_labels = get_mean_for_each_label(x_train, y_train)
    train_data_len = len(x_train)
    generators = [
        Generator(
            in_features=train_cfg.LATENT_SIZE,
            out_features=len(_features),
            hidden_layers=model.G_HIDDEN_LAYERS,
        ) for _ in range(4)
    ]
    use_mean_std = [False, True, False, True]
    number_of_data = [train_data_len * i for i in (2, 5, 20, 50, 100)]
    for gen_i, gen in enumerate(generators):
        gen.load_state_dict(torch.load(generator_paths[gen_i][1]))
        gen.eval()
        print(f'Generator: {generator_paths[gen_i][0]}')

        for cls_name, cls in classifiers:

            for data_len in number_of_data:
                if use_mean_std[gen_i]:
                    fake_train, fake_train_class = generate_data(gen, data_len, unique_labels, label_means, label_stds)
                    # fake_test, fake_test_class = generate_data(gen, test_data_len, unique_labels, label_means, label_stds)
                else:
                    fake_train, fake_train_class = generate_data(gen, data_len)
                    # fake_test, fake_test_class = generate_data(gen, test_data_len)

                model_calib = ModelCalibration(
                    cls,
                    x_train=fake_train,
                    y_train=fake_train_class.ravel(),
                    x_test=x_test,
                    y_test=y_test.ravel(),
                    name=generator_paths[gen_i][0] + cls_name + f"_{data_len}",
                )
                model_calib.fit()
                # plot_calibration_curve(model_calib.y_test, model_calib.uncalibrate_result, model_calib.name + '_uncalib',
                #                        show=False)
                # plot_calibration_curve(model_calib.y_test, model_calib.platt_result_sklearn,
                #                        model_calib.name + '_sklearn calib', show=False)
                print('-----------------', model_calib.name, '-----------------')
                bootstrapping_brier_score(model_calib.model, x_test, y_test.ravel(), 400)
                calculate_confidence_interval(model_calib.model, x_test, y_test.ravel(), 400)
                bootstrapping_f1_score(model_calib.model, x_test, y_test.ravel(), 400)


def train_with_original_data(models: list, train_data_path, test_data_path, _features):
    train_dataset = GenomeDataset(dt_cfg, train_data_path, _features)
    x_train = train_dataset.scaler.fit_transform(train_dataset.x)
    y_train = train_dataset.y.ravel()

    val_dataset = GenomeDataset(dt_cfg, test_data_path, _features)
    x_test = val_dataset.scaler.fit_transform(val_dataset.x)
    y_test = val_dataset.y.ravel()

    for name, m in models:
        if isinstance(m, Path):
            pretrained_model = pickle.load(open(m, 'rb'))
            model_calib = ModelCalibration(
                pretrained_model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train.astype(np.int64),
                y_test=y_test.astype(np.int64),
                name=os.path.basename(name).split('.')[0],
                is_trained=True
            )
        else:
            model_calib = ModelCalibration(
                m,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train.astype(np.int64),
                y_test=y_test.astype(np.int64),
                name=os.path.basename(name).split('.')[0],
            )
        model_calib.fit()
        model_calib.save_models()
        print(name, 'Uncalibrate brier score: ', calculate_brier_score(model_calib.y_test, model_calib.uncalibrate_result))
        print(name, 'Calibrate brier score: ', calculate_brier_score(model_calib.y_test, model_calib.platt_result_sklearn))
        plot_calibration_curve(model_calib.y_test, model_calib.uncalibrate_result, model_calib.name + '_uncalib',
                               show=False)
        plot_calibration_curve(model_calib.y_test, model_calib.platt_result_sklearn,
                               model_calib.name + '_sklearn calib', show=False)


if __name__ == '__main__':
    test_data_path = Path('dataset/8_202_test.csv')
    train_data_path = Path('dataset/8_202_train.csv')
    selected_features = features.SELECTED_FEATURES_64_8_202
    model_name_prefix = '_8_202_Generated'

    generator_paths = [
        ('G_no_mean_std_8_202', Path("SavedModels/2022-12-02 18_21_53acgan_without_mean_std_8_202.pth")),
        ('G_mean_std_8_202', Path("SavedModels/2022-12-02 18_25_32acgan_mean_std_8_202.pth")),
        ('Basic_G_no_mean_std_8_202', Path("SavedModels/2022-12-02 18_31_38basic_gan_without_mean_std_8_202.pth")),
        ('Basic_G_mean_std_8_202', Path("SavedModels/2022-12-02 18_33_10basic_gan_with_mean_std_8_202.pth")),
    ]
    validation_model_paths = [
        Path('SavedModels/2022-11-24 23_27_11_LR_new_features.pickle'),
        Path('SavedModels/2022-11-24 23_27_14_NN_new_features.pickle'),
        Path('SavedModels/2022-11-24 23_27_18_1dcnn_new_features.pickle'),
        Path('SavedModels/2022-11-24 23_27_20_knn_new_features.pickle'),
        Path('SavedModels/2022-11-24 23_27_21_svm_new_features.pickle'),
    ]
    classifiers = make_classification_algorithms(len(selected_features), model_name_prefix)
    # train_with_original_data(
    #     models=classifiers,
    #     train_data_path=train_data_path,
    #     test_data_path=test_data_path,
    #     _features=selected_features,
    # )
    train_with_generated_data(generator_paths, classifiers, train_data_path, test_data_path, selected_features)
