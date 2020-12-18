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
    pipeline = Pipeline(steps)
    parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
    grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    OK = True
    C = 1
    is_labeled = [True] * label_init_size + [False] * (ll-label_init_size)

    for i in [0.0001,0.0002,0.0005,0.0007,0.002,0.005,0.007]:
        # print(10**(-i))
        OK = True
        C = i
        is_labeled = [True] * label_init_size + [False] * (ll - label_init_size)

        parameters = {'SVM__C': [i], 'SVM__gamma': [10, 1, 0.1, 0.01]}
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

        while OK:
            print(C)
            w = [C] * label_init_size + [C/2] * (ll - label_init_size)  # TODO
            X_train = list(compress(X_data, is_labeled))
            y_train = list(compress(y_data, is_labeled))

            unlabeled = list(compress(X_data, [not b for b in is_labeled]))
            w_train = list(compress(w, is_labeled))

            print(len(X_train))
            if len(X_train) == ll:
                break
            grid.fit(X_train, y_train, w_train)

            print("trained")
            C = grid.best_params_["SVM__C"]
            # g = grid.best_params_["SVM__gamma"]

            # print("score = %3.2f" % (grid.score(X_test, y_test)))

            # print("best parameters from train data: ", grid.best_params_)


            unlabeled_pred2 = grid.predict(unlabeled)

            it = 0
            a = 0
            b = 0
            OK = False

            print(a, b)
            print("score = %3.2f" % (grid.score(X_test, y_test)))
            # print(type(y_pred))
            # print(len(y_pred))

            # print(y_pred[100:105])
            # print(y_test[100:105])

        y_pred = grid.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        print("confusion matrix: \n ", cm)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        print("TPR", TPR)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        print("TNR", TNR)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        print("PPV", PPV)
        # Negative predictive value
        NPV = TN / (TN + FN)
        print("NPV", NPV)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        print("FPR", FPR)
        # False negative rate
        FNR = FN / (TP + FN)
        print("FNR", FNR)
        # False discovery rate
        FDR = FP / (TP + FP)
        print("FDR", FDR)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        print("ACC", ACC)

        print("\n\n\n\n")
    endt = time.time()

    # print("total time taken = %3.3f" % (endt - start))
    # ss = np.sum(cm, axis=0)
    # print(ss)
    # ss = np.sum(cm, axis=1)
    # print(ss)
