import sklearn
import numpy as np
import math, time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from itertools import compress
from sklearn.decomposition import PCA
import collections

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
    ratio = 0.4
    label_init_size = int(ll*ratio)
    # labeled_data = dataset['train_x'][0:int(ratio*len(dataset['train_x'])), :]
    # labels = dataset['train_y'][0:int(ratio*len(dataset['train_y']))]
    # print(labels)

    start = time.time()

    X_data = dataset['train_x']
    y_data = dataset['train_y']

    X_test = dataset['test_x']
    y_test = dataset['test_y']

    steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='linear', probability=True))]
    pipeline = Pipeline(steps)
    # parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
    parameters = {'SVM__C': [0.001], 'SVM__gamma': [10]}
    grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    OK = True
    C = 1
    is_labeled = [True] * label_init_size + [False] * (ll-label_init_size)

    pca = PCA()
    pca.fit(X_data)
    score = pca.transform(X_data)
    print(pca.explained_variance_ratio_)


    # for i in [0.0001,0.0002,0.0005,0.0007,0.002,0.005,0.007]:
    for iii in [0.0001]:
        # print(10**(-i))
        OK = True
        C = iii
        is_labeled = [True] * label_init_size + [False] * (ll - label_init_size)

        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        # parameters = {'SVM__C': [iii], 'SVM__gamma': [1]}
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    endt = time.time()

    # print("total time taken = %3.3f" % (endt - start))
    # ss = np.sum(cm, axis=0)
    # print(ss)
    # ss = np.sum(cm, axis=1)
    # print(ss)
