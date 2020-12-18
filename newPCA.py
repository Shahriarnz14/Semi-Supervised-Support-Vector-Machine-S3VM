import sklearn
import numpy as np
import math, time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from itertools import compress
from sklearn.decomposition import PCA
import collections

from sklearn.manifold import TSNE
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

    print("hello")
    dataset = load_data()
    ll = len(dataset['train_x'])
    ratio = 3000.0 / 3000.0
    label_init_size = int(ll*ratio)
    # labeled_data = dataset['train_x'][0:int(ratio*len(dataset['train_x'])), :]
    # labels = dataset['train_y'][0:int(ratio*len(dataset['train_y']))]
    # print(labels)

    start = time.time()

    X_data = dataset['train_x']
    y_data = dataset['train_y']

    X_test = dataset['test_x']
    y_test = dataset['test_y']

    print("here2")

    steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='linear', probability=True))]
    pipeline = Pipeline(steps)

    pca = PCA(200)
    pca_full = pca.fit(X_data)
    score = pca.transform(X_data)

    t_pca = pca.transform(X_test)
    testa = [s[::2] for s in t_pca]
    testb = [s[1::2] for s in t_pca]
    testk = t_pca

    print("PCA")
    #

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(score[:3000, 0], score[:3000, 1], c=y_data[:3000], edgecolor='none', alpha=0.5, cmap=plt.get_cmap('jet', 10), s=5)
    plt.colorbar()
    plt.xlabel('Principal Component #0', fontsize=14)
    plt.ylabel('Principal Component #1', fontsize=14)
    plt.title('(a) PCA Plot of MNIST Digits Dataset', fontsize=16)
    #plt.xlim(-5, 10)
    #plt.ylim(-3, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()

    print("plotted PCA")

    X_embedded = TSNE(n_components=2).fit_transform(X_data)

    plt.subplot(122)
    plt.scatter(X_embedded[:, 1], X_embedded[:, 0], c=y_data, alpha=0.5, cmap=plt.get_cmap('jet', 10), s=5)
    plt.colorbar()
    plt.xlabel('t-SNE Dimension #0', fontsize=14)
    plt.ylabel('t-SNE Dimension #1', fontsize=14)
    plt.title('(b) t-SNE Plot of MNIST Digits Dataset', fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    OK = True
    is_labeled = [True] * label_init_size + [False] * (ll-label_init_size)

    while OK:

        parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        grida = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        gridb = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        gridk = GridSearchCV(pipeline, param_grid=parameters, cv=5)

        # print("here")
        scorea = [s[::2] for s in score]
        scoreb = [s[1::2] for s in score]

        X_traina = list(compress(scorea, is_labeled))
        X_trainb = list(compress(scoreb, is_labeled))
        X_traink = list(compress(score, is_labeled))
        y_train = list(compress(y_data, is_labeled))

        unlabeleda = list(compress(scorea, [not b for b in is_labeled]))
        unlabeledb = list(compress(scoreb, [not b for b in is_labeled]))
        unlabeledk = list(compress(score, [not b for b in is_labeled]))
        unlabeled_ind = list(compress(range(len(X_data)), [not b for b in is_labeled]))
        print(unlabeled_ind)

        print(len(X_traink))
        print(len(X_traink[0]))
        counter = collections.Counter(y_train)
        print(counter)

        gridk.fit(X_traink, y_train)

        print("score = %3.2f" % (gridk.score(testk, y_test)))

        grida.fit(X_traina, y_train)

        print("score = %3.2f" % (grida.score(testa, y_test)))

        gridb.fit(X_trainb, y_train)

        print("score = %3.2f" % (gridb.score(testb, y_test)))

        if len(X_traina) == ll or not OK:
            break

        print("trained")
        probsa = grida.predict_proba(unlabeleda)
        probsb = gridb.predict_proba(unlabeledb)
        unlabeled_preda = grida.predict(unlabeleda)
        unlabeled_predb = gridb.predict(unlabeledb)

        OK = False

        a = b = 0
        for i in range(10):

            tv = probsa[unlabeled_preda == i]
            tv_ind = list(compress(range(len(probsa)), [unlabeled_preda == i]))
            tmax = [v[i] for v in tv]
            tresh = sorted(tmax)[-4]

            for j in range(len(tv)):
                if tmax[j] > tresh:
                    it = tv_ind[j]

                    a += 1
                    if y_data[it] != i:
                        b += 1

                    is_labeled[it] = True
                    y_data[it] = i
                    OK = True

        print(a, b)

        a = b = 0
        for i in range(10):

            tv = probsb[unlabeled_predb == i]
            tv_ind = list(compress(range(len(probsb)), [unlabeled_predb == i]))
            tmax = [v[i] for v in tv]
            tresh = sorted(tmax)[-4]

            for j in range(len(tv)):
                if tmax[j] > tresh:
                    it = tv_ind[j]

                    a += 1
                    if y_data[it] != i:
                        b += 1

                    is_labeled[it] = True
                    y_data[it] = i
                    OK = True
        print(a, b)
