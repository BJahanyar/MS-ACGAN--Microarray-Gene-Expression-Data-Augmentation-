from pprint import pprint
import os
import pickle
import typing
from pathlib import Path

import torch
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skorch import NeuralNetClassifier

from config import train as train_cfg, features, model, dataset as dt_cfg
from dataloader import GenomeDataset
from models import MLPClassifier, CNN1Classifier
from utils.utils import load_train_test_data, current_time
from utils.train_utils import get_mean_for_each_label
from models.generator import Generator


class Metrics(BaseModel):
    train_accuracy: float = None
    train_loss: float = None
    train_f1_score: float = None
    train_confusion_matrix: list = None
    val_accuracy: float = None
    val_loss: float = None
    val_f1_score: float = None
    val_confusion_matrix: list = None


class Classification:
    def __init__(self, name: str, save_model: bool = True):
        self.name = name
        self._algorithm = None
        self._train_data = None
        self._val_data = None

        self._save_model = save_model
        self._metrics = Metrics()

    def set_algorithm(self, algorithm):
        assert hasattr(algorithm, 'fit')
        assert hasattr(algorithm, 'score')
        assert hasattr(algorithm, 'predict')
        self._algorithm = algorithm

    def set_dataset(
            self,
            train_data: typing.Tuple[np.ndarray, np.ndarray],
            val_data: typing.Tuple[np.ndarray, np.ndarray] = None,
    ):
        """

        :param train_data: (x_train, y_train)
        :param val_data: (x_val, y_val)
        """
        self._train_data = train_data
        self._val_data = val_data

    def fit(self):
        assert self._algorithm is not None, 'Algorithm not defined'
        assert self._train_data is not None, 'Train data not defined'
        self.train()
        self.validation()
        self.print_metrics()
        self.save_model()

    def train(self):
        self._algorithm.fit(self._train_data[0], self._train_data[1].ravel())
        y_predict = self._algorithm.predict(self._train_data[0])
        self._metrics.train_accuracy = accuracy_score(self._train_data[1], y_predict)
        self._metrics.train_loss = log_loss(self._train_data[1], y_predict)
        self._metrics.train_f1_score = f1_score(self._train_data[1], y_predict)
        self._metrics.train_confusion_matrix = confusion_matrix(self._train_data[1], y_predict).tolist()

    def validation(self):
        if self._val_data is None:
            print('No Validation data available')
            return

        y_predict = self._algorithm.predict(self._val_data[0])
        self._metrics.val_accuracy = accuracy_score(self._val_data[1], y_predict)
        self._metrics.val_loss = log_loss(self._val_data[1], y_predict)
        self._metrics.val_f1_score = f1_score(self._val_data[1], y_predict)
        self._metrics.val_confusion_matrix = confusion_matrix(self._val_data[1], y_predict).tolist()

    def print_metrics(self):
        print('*' * 20, self.name, '*' * 20)
        pprint(self._metrics.dict(), indent=4)

    def save_model(self):
        if not self._save_model:
            return
        model_path = os.path.join(model.SAVE_PATH, current_time() + '_' + self.name + '.pickle')
        pickle.dump(
            self._algorithm,
            open(model_path, 'wb')
        )
        print(f'Model saved: {model_path}')


def validation_with_cls(cls, x_test, y_test):
    test_score = cls.score(x_test, y_test)
    y_pred = cls.predict(x_test)
    print(f"Test score: {test_score:.4f}")
    print(f"confusion matrix:\n{confusion_matrix(y_test, y_pred)}")


def logistic_regression_classifier():
    x_train, y_train, x_test, y_test = load_train_test_data(train_cfg.TRAIN_DATA_PATH,
                                                            train_cfg.TEST_DATA_PATH,
                                                            features.SELECTED_FEATURES_64_8_202)

    lr = LogisticRegression(max_iter=1000)
    cls = Classification(model.LR_VALIDATION_MODEL, save_model=True)
    cls.set_dataset((x_train, y_train), (x_test, y_test))
    cls.set_algorithm(lr)
    cls.fit()


def nn_classifier():
    x_train, y_train, x_test, y_test = load_train_test_data(train_cfg.TRAIN_DATA_PATH,
                                                            train_cfg.TEST_DATA_PATH,
                                                            features.SELECTED_FEATURES_64_8_202)

    net = NeuralNetClassifier(
        MLPClassifier,
        max_epochs=train_cfg.EPOCHS,
        lr=0.01,
        iterator_train__shuffle=True,
        module__input_features=x_train.shape[1],
        module__num_classes=2,
        module__hidden_layers=[32],
        verbose=0,
    )

    cls = Classification(model.NN_VALIDATION_MODEL, save_model=True)
    cls.set_dataset((x_train, y_train), (x_test, y_test))
    cls.set_algorithm(net)
    cls.fit()


def grid_search_nn_classifier():
    dataset = GenomeDataset(dt_cfg, train_cfg.TRAIN_DATA_PATH, features.MRMR_SELECTED_FEATURES_10)
    x = dataset.scaler.fit_transform(dataset.x)
    y = dataset.y
    y = y.astype(np.int64).ravel()

    net = NeuralNetClassifier(
        MLPClassifier,
        max_epochs=train_cfg.EPOCHS,
        lr=0.01,
        iterator_train__shuffle=True,
        module__input_features=dataset.data_size,
        module__num_classes=dataset.number_of_classes,
        module__hidden_layers=[128],
        train_split=False,
    )
    params = {
        'lr': [0.1, 0.01, 0.001],
        'module__hidden_layers': [[128], [128, 128], [128, 128, 128],
                                  [16], [16, 16], [16, 16, 16],
                                  [32], [32, 32], [32, 32, 32],
                                  [32, 16], [32, 16, 4], [4, 4, 4],
                                  [4, 16, 32], [128, 16, 4]]
    }
    gs = GridSearchCV(net, params, refit=False, cv=5, scoring='accuracy', verbose=2, n_jobs=4)
    gs.fit(x, y)
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))


def cnn1d_classifier():
    x_train, y_train, x_test, y_test = load_train_test_data(train_cfg.TRAIN_DATA_PATH,
                                                            train_cfg.TEST_DATA_PATH,
                                                            features.SELECTED_FEATURES_64_8_202)
    # x_train = np.expand_dims(x_train, axis=1)
    # x_test = np.expand_dims(x_test, axis=1)
    net = NeuralNetClassifier(
        CNN1Classifier,
        max_epochs=train_cfg.EPOCHS,
        lr=0.1,
        iterator_train__shuffle=True,
        module__input_channel=1,
        module__num_classes=2,
        module__hidden_layers=[64],
        verbose=0,
    )

    cls = Classification(model.CNN1D_VALIDATION_MODEL, save_model=True)
    cls.set_dataset((x_train, y_train), (x_test, y_test))
    cls.set_algorithm(net)
    cls.fit()


def knn_classifier():
    x_train, y_train, x_test, y_test = load_train_test_data(train_cfg.TRAIN_DATA_PATH,
                                                            train_cfg.TEST_DATA_PATH,
                                                            features.SELECTED_FEATURES_64_8_202)

    cls = Classification(model.KNN_VALIDATION_MODEL, save_model=True)
    cls.set_dataset((x_train, y_train), (x_test, y_test))
    cls.set_algorithm(KNeighborsClassifier())
    cls.fit()


def svm_classifier():
    x_train, y_train, x_test, y_test = load_train_test_data(train_cfg.TRAIN_DATA_PATH,
                                                            train_cfg.TEST_DATA_PATH,
                                                            features.SELECTED_FEATURES_64_8_202)

    cls = Classification(model.SVC_VALIDATION_MODEL, save_model=True)
    cls.set_dataset((x_train, y_train), (x_test, y_test))
    cls.set_algorithm(SVC())
    cls.fit()


def train_classifiers_with_fake_data():
    x_train, y_train, x_test, y_test = load_train_test_data(train_cfg.TRAIN_DATA_PATH,
                                                            train_cfg.TEST_DATA_PATH,
                                                            features.SELECTED_FEATURES_64_8_202)

    label_means, label_stds, unique_labels = get_mean_for_each_label(x_train, y_train)
    train_data_len = len(x_train)
    classifiers = [
        LogisticRegression(max_iter=1000),
        NeuralNetClassifier(
            MLPClassifier,
            max_epochs=100,
            lr=0.01,
            iterator_train__shuffle=True,
            module__input_features=x_train.shape[1],
            module__num_classes=2,
            module__hidden_layers=[32],
            verbose=0,
        ),
        NeuralNetClassifier(
            CNN1Classifier,
            max_epochs=100,
            lr=0.1,
            iterator_train__shuffle=True,
            module__input_channel=1,
            module__num_classes=2,
            module__hidden_layers=[64],
            verbose=0,
        ),
        KNeighborsClassifier(),
        SVC(),
    ]
    generators = [
        Generator(
            in_features=train_cfg.LATENT_SIZE,
            out_features=len(features.SELECTED_FEATURES_64_8_202),
            hidden_layers=model.G_HIDDEN_LAYERS,
        ) for _ in range(4)
    ]
    generator_paths = [
        Path("SavedModels/2022-12-02 18_21_53acgan_without_mean_std_8_202.pth"),
        Path("SavedModels/2022-12-02 18_25_32acgan_mean_std_8_202.pth"),
        Path("SavedModels/2022-12-02 18_31_38basic_gan_without_mean_std_8_202.pth"),
        Path("SavedModels/2022-12-02 18_33_10basic_gan_with_mean_std_8_202.pth"),
    ]
    use_mean_std = [False, True, False, True]
    number_of_data = [train_data_len * i for i in (2, 5, 20, 50, 100)]

    for gen_i, gen in enumerate(generators):
        gen.load_state_dict(torch.load(generator_paths[gen_i]))
        gen.eval()
        print(f'Generator: {generator_paths[gen_i]}')
        for cls in classifiers:
            for data_len in number_of_data:
                print(f"data length {data_len}")
                if use_mean_std[gen_i]:
                    fake_train, fake_class = generate_data(gen, data_len, unique_labels, label_means, label_stds)
                else:
                    fake_train, fake_class = generate_data(gen, data_len)

                make_cls = Classification(str(cls), save_model=False)
                make_cls.set_algorithm(cls)
                make_cls.set_dataset((fake_train, fake_class), (x_test, y_test))
                make_cls.fit()


def generate_data(gen, data_length, unique_labels=None, mean=None, std=None):
    noise = torch.randn(data_length, train_cfg.LATENT_SIZE, dtype=torch.float32)
    fake_classes = torch.randint(0, 2, (data_length, 1), dtype=torch.float32)
    discrete_noise = noise + fake_classes
    if mean and std:
        for ii, l in enumerate(unique_labels):
            noise[fake_classes.squeeze(1) == l] = (noise[fake_classes.squeeze(1) == l] + mean[ii]) * \
                                                  std[ii]
        with torch.no_grad():
            generated_data = gen(noise)
        return generated_data.cpu().numpy(), fake_classes.cpu().numpy().astype(np.int64)

    with torch.no_grad():
        generated_data = gen(discrete_noise)
    return generated_data.cpu().numpy(), fake_classes.cpu().numpy().astype(np.int64)
