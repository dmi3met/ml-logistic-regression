import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score

data = pd.read_csv('data-logistic.csv', header=None)
# print(data)
X_ = data.iloc[:, 1:]
y_ = data.iloc[:, 0]


def sigmoid(w1, x1, w2, x2):
    return 1.0 / (1 + np.exp(- w1 * x1 - w2 * x2))
    # 1.0 / (1 + math.exp(-x))


def distance(a, b):
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))  # ok


def log_regression(X, y, k, w, C, epsilon, max_iter):
    w1, w2 = w
    x1 = X.iloc[:, 0]
    x2 = X.iloc[:, 1]

    l = len(x1)
    for n in range(max_iter):
        summ_xi1 = summ_xi2 = 0
        for i in range(l):
            b = (1 - (1.0 / (1 + np.exp(-y[i] * (w1 * x1[i] + w2 * x2[i])))))
            summ_xi1 += y[i] * x1[i] * b
            summ_xi2 += y[i] * x2[i] * b

        w1new = w1 + k / l * summ_xi1 - k * C * w1
        w2new = w2 + k / l * summ_xi2 - k * C * w2

        if distance((w1new, w2new), (w1, w2)) < epsilon:  # ok
            break
        w1, w2 = w1new, w2new


    predictions = []

    for i in range(l):
        # t1 = -w1 * X[i, 0] - w2 * X[i, 1]
        s = sigmoid(w1, x1[i], w2, x2[i])
        predictions.append(s)
    return predictions


p0 = log_regression(X_, y_, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
p1 = log_regression(X_, y_, 0.1, [0.0, 0.0], 10, 0.00001, 10000)
print(np.shape(y_))
print(np.shape(p0))
print(np.shape(p1))
print(roc_auc_score(y_, p0))
print(roc_auc_score(y_, p1))

