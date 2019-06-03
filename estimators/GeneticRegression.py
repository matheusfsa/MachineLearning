import numpy as np
from core import *
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

class GeneticRegression(object):
    def __init__(self, max_iterations=100000, learning_ratio=0.3, tol=0.0001, pop_size=100):
        self.w = np.array([])
        self.b = 0
        self.is_fit = False
        self.max_iterations = max_iterations
        self.learning_ratio = learning_ratio
        self.tol = tol
        self.i = 0
        self.pop_size = pop_size
        self.erro = 0

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.i = 0
        population = np.random.rand(self.pop_size, X.shape[1])
        fitness = self.evaluate(population, X, y)
        while self.i < self.max_iterations and (np.min(population, axis=0) != np.max(population, axis=0)).all():
            offspring_population = self.reproduction(population)
            fitness_offspring = self.evaluate(offspring_population, X, y)
            population, fitness = self.selection(population, fitness, offspring_population, fitness_offspring)
            self.w = population[0]
            self.erro = fitness[0]
            self.i += 1
        self.is_fit = True
        return self

    def selection(self, PX, PY, QX, QY):
        RX, RY = np.append(PX, QX, axis=0), np.append(PY, QY, axis=0)
        args = (np.abs(RY)).argsort()
        RX = RX[args]
        RY = RY[args]
        return RX[:self.pop_size], RY[:self.pop_size]

    def evaluate(self, W, X, y):

        y_p = np.dot(W, X.T)
        left = np.multiply(y, np.log(y_p))
        right = np.multiply(1 - y, np.log(y_p))
        res = np.sum(left + right, axis=1)
        return res

    def reproduction(self, w):
        WX = np.empty((0, w.shape[1]))
        lower, upper = np.min(w, axis=0), np.max(w, axis=0)
        for i in range(0, w.shape[0], 2):
            cx1, cx2 = one_point_crossover(w[i], w[i + 1])
            cx1 = mutation(cx1)
            cx2 = mutation(cx2)
            WX = np.append(WX, [cx1, cx2], axis=0)
        print(WX)
        return WX

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        if self.is_fit:
            z = np.dot(self.w, X.T)
            res = self.sigmoid(z)
            res[res >= 0.5] = 1.0
            res[res < 0.5] = 0.0
            return res
        raise RuntimeError('This LogisticRegression instance is not fitted yet.'
                           ' Call fit with appropriate arguments before using this method.')


X = np.array([[1, 1], [0, 0]])
y = np.array([1, 0])
data = load_iris()
X = data.data[:100]
y = data.target[:100]
gen = GeneticRegression()
gen.fit(X, y)
preds = gen.predict(X)