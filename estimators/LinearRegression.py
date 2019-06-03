import numpy as np
"""
Seção 5.1.4 do livro Deep Learning de Ian Goodfellow e Yoshua Bengio e Aaron Courville, 
utilizando equação 5.12.

"""


class LinearRegression(object):
    def __init__(self):
        self.w = np.array([])
        self.is_fit = False

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        Xt = np.transpose(X)
        left = np.linalg.inv(np.dot(Xt, X))
        right = np.dot(Xt, y)
        self.w = np.dot(left, right)
        self.is_fit = True
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        print(X)
        if self.is_fit:
            return np.dot(np.transpose(self.w[0]), X)
        return None


def test(X, y, Xtest, Ytest):

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(Xtest)
    print(y_pred - Ytest)


X = np.array([[0.2], [0.6]])
y = np.array([[1], [3]])
Xtest = np.array([[0.4]])
Ytest = np.array([[0.7]])

test(X, y, Xtest, Ytest)



