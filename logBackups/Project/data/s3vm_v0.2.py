import sklearn
import numpy as np
import math, time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from itertools import compress

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    data_train = np.genfromtxt('data/train.csv', dtype=np.float64, delimiter=',')
    assert data_train.shape == (3000, 785)
    data_test = np.genfromtxt('data/test.csv', dtype=np.float64, delimiter=',')

    assert data_test.shape == (1000, 785)

    return {
        'train_x': data_train[:, :-1].astype(np.float64),
        'train_y': data_train[:, -1].astype(np.int64),
        'test_x': data_test[:, :-1].astype(np.float64),
        'test_y': data_test[:, -1].astype(np.int64),
    }


if __name__ == '__main__':

    dataset = load_data()
    ll = len(dataset['train_x'])
    ratio = 0.2
    label_init_size = int(ll*ratio)
    # labeled_data = dataset['train_x'][0:int(ratio*len(dataset['train_x'])), :]
    # labels = dataset['train_y'][0:int(ratio*len(dataset['train_y']))]
    # print(labels)

    start = time.time()

    X_data = dataset['train_x']
    y_data = dataset['train_y']

    X_test = dataset['test_x']
    y_test = dataset['test_y']

    steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='linear'))]
    parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
    pipeline = Pipeline(steps)
    grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    OK = True
    C = 1
    is_labeled = [True] * label_init_size + [False] * (ll-label_init_size)

    while OK:

        w = [C] * label_init_size + [C / 2] * (ll - label_init_size)  # TODO
        X_train = list(compress(X_data, is_labeled))
        y_train = list(compress(y_data, is_labeled))

        unlabeled = list(compress(X_data, [not b for b in is_labeled]))
        w_train = list(compress(w, is_labeled))

        print(len(X_train))
        grid.fit(X_train, y_train, w_train)

        print("trained")
        C = grid.best_params_["SVM__C"]
        # g = grid.best_params_["SVM__gamma"]

        # print("score = %3.2f" % (grid.score(X_test, y_test)))

        # print("best parameters from train data: ", grid.best_params_)

        # y_pred = grid.predict(X_test)
        unlabeled_pred = grid.predict(unlabeled)
        print("predicted")

        print("score = %3.2f" % (grid.score(X_test, y_test)))

        grid.fit(X_train+unlabeled, y_train+unlabeled_pred.tolist(), w)

        print("trained 2")

        unlabeled_pred2 = grid.predict(unlabeled)

        print("predicted 2")

        it = 0
        a = 0
        b = 0
        OK = False
        for i in range(len(unlabeled_pred)):
            while is_labeled[it]:
                it += 1
            if unlabeled_pred[i] == unlabeled_pred2[i]:
                a += 1
                if y_data[it] != unlabeled_pred[i]:
                    b += 1
                is_labeled[it] = True
                y_data[it] = unlabeled_pred[i]
                OK = True
            it += 1

        print(a, b)
        print("score = %3.2f" % (grid.score(X_test, y_test)))
        # print(type(y_pred))
        # print(len(y_pred))

        # print(y_pred[100:105])
        # print(y_test[100:105])

        # cm = confusion_matrix(y_test, y_pred)
        # print("confusion matrix: \n ", cm)

    endt = time.time()

    # print("total time taken = %3.3f" % (endt - start))
    # ss = np.sum(cm, axis=0)
    # print(ss)
    # ss = np.sum(cm, axis=1)
    # print(ss)
