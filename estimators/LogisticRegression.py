import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from pimg import *

class LogisticRegression(object):

    def __init__(self, max_iterations=1000, learning_ratio=0.3, tol=0.0001):
        self.w = np.array([])
        self.b = 0
        self.is_fit = False
        self.max_iterations = max_iterations
        self.learning_ratio = learning_ratio
        self.tol = tol
        self.i = 0

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        self.i = 0
        self.w = np.random.rand(1, X.shape[1])
        self.b = 0
        alpha = self.learning_ratio/X.shape[0]
        diff = np.inf
        while self.i < self.max_iterations and diff > self.tol:
            z = np.dot(self.w, X.T) + self.b
            delta = self.sigmoid(z) - y.T
            w_ant = np.copy(self.w)
            print(delta.shape)
            print(X.shape)
            dw = np.dot(delta, X)
            print(dw.shape)
            self.w = w_ant - alpha*(dw)

            self.b = self.b - alpha*(np.sum(delta))
            diff = np.linalg.norm(w_ant-self.w, np.inf)
            self.i += 1
        self.is_fit = True
        return self

    def predict(self, X):
        if self.is_fit:
            z = np.dot(self.w, X.T) + self.b
            res = self.sigmoid(z)
            res[res >= 0.5] = 1.0
            res[res < 0.5] = 0.0
            return res
        raise RuntimeError('This LogisticRegression instance is not fitted yet.'
                           ' Call fit with appropriate arguments before using this method.')


data = load_iris()
X = data.data[:100]
y = data.target[:100]
clf = LogisticRegression(max_iterations=1)


