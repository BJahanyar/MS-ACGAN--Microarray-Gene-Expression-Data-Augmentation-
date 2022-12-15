from pprint import pprint
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate


def multiple_model_training(opt: dict, x: np.ndarray, y: np.ndarray):
    models = [
        ("SVC", SVC()),
        ("LogisticRegression", LogisticRegression()),
        ("KNeighborClassifier", KNeighborsClassifier()),
        ("NeuralNetwork",
         MLPClassifier(hidden_layer_sizes=opt['hidden_layer_size'],
                       solver='adam',
                       activation='relu',
                       batch_size=opt['batch_size'],
                       learning_rate='invscaling',
                       learning_rate_init=opt['learning_rate'],
                       max_iter=opt['neural_network_max_iter']))
    ]
    kf = KFold(n_splits=opt["n_splits"], shuffle=True)
    for name, model in models:
        cv_result = cross_validate(model, x, y.ravel(), cv=kf)
        print(name)
        pprint(cv_result, indent=4)
