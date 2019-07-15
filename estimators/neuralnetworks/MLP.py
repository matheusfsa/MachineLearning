import numpy as np
from estimators.neuralnetworks.layers import *
from sklearn.datasets import load_iris, load_wine
import matplotlib.pyplot as plt
class MLP:
    def __init__(self, layers, epochs=100, learning_ratio=0.03, cost='log', tol=1e-6, batch_size=10):
        self.layers = layers
        self.l = len(layers)
        self.a = None
        self.Zs = []
        self.cost = cost
        self.learning_ratio = learning_ratio
        self.epochs = epochs
        self.errors = []
        self.score = 0.0
        self.n_iteractions = 0
        self.tol = tol
        self.batch_size = batch_size

    def load_net(self, n):
        self.layers[0].initialize_parameters(n)
        for i in range(1, len(self.layers)):
            self.layers[i].initialize_parameters(self.layers[i-1].units)

    def loss_function(self, name):
        if name == 'cross-entropy':
            return lambda y_true, y_pred: (-y_true*np.log(y_pred) - (1 - y_true)*np.log(1-y_pred)).mean()

    def foward_propagation(self, X):
        self.Zs = []
        A = np.copy(X)
        for layer in self.layers:
            A = layer.activation(A)
            self.Zs.append(layer.z)
        return A

    def backward_propagation(self, y):
        da = self.layers[self.l-1].get_da(y)
        for i in range(self.l-1, 0, -1):
            da = self.layers[i].backward(da, self.learning_ratio)

    def predict(self, X):
        A = self.foward_propagation(X)
        A[A >= 0.5] = 1
        A[A < 0.5] = 0
        return A

    def update_weights(self, y):
        for i in range(self.l):
            self.layers[i].update(self.learning_ratio, y.shape[0])

    def getDerivitiveActivationFunction(self, name):
        if name == 'sigmoid':
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        if name == 'linear':
            return lambda x: 1
        if name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y

            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1

    def error_derivative(self, name):
        if name == 'log':

            return lambda y, a: (y - a)
        if name == 'quadratic':
            return lambda y, a: y - a

    def calculate_output_error(self, y, cost_name, layer):
            grad_cost = self.error_derivative(cost_name)(y.reshape((-1, 1)), layer.A)
            derivative_activation = layer.derivative_activation(layer.z)
            return np.multiply(grad_cost, derivative_activation)

    def get_error(self, X, y):
        epsilon = 1e-12
        y_pred = self.foward_propagation(X)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        erro = self.loss_function('cross-entropy')(y, y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        self.score = np.sum(y_pred == y) / y.shape[0]
        return erro

    def compute_derivative(self, y):
        self.backpropagation(y)
        self.update_weights(y)

    def fit(self, X, y):
        n = X.shape[0]
        y = y.reshape(-1, 1)
        self.load_net(X.shape[1])
        self.errors = []
        self.init_parameters()
        self.n_iteractions = 0
        self.score = 0
        delta = np.inf
        i = 0
        k = (n // self.batch_size)
        while i < self.epochs and delta > self.tol:
            ini = 0
            fim = self.batch_size
            X_train = X[ini:fim, :]
            y_train = y[ini:fim, :]
            erro_ant = self.get_error(X_train, y_train)
            epoch_ended = False
            erro = 0
            while not epoch_ended:
                self.compute_derivative(y_train)
                ini = fim
                fim += self.batch_size
                if fim > n:
                    fim = n
                    epoch_ended = True
                if ini < n:
                    X_train = X[ini:fim, :]
                    y_train = y[ini:fim, :]
                    erro += self.get_error(X_train, y_train)/k
            delta = np.abs(erro_ant - erro)
            erro_ant = erro
            print('Epochs:', i, ' Error:', erro)
            i += 1
        '''
        epsilon = 1e-12
        delta = np.inf
        i=0
        while i < self.epochs and self.score < 1.0 and delta > self.tol:
            y_pred = self.foward_propagation(X)
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            erro = self.loss_function('cross-entropy')(y, y_pred)
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            self.score = np.sum(y_pred == y)/y.shape[0]
            self.errors.append(erro)
            self.backpropagation(y)
            self.update_weights(y)
            if i < 0:
                delta = np.abs(self.errors[i] - self.errors[i-1])
            print('Epochs:', i, ' Error:', erro)
            i += 1
        self.n_iteractions = i
        '''
    def init_parameters(self):
        for layer in self.layers:
            layer.restart()

    def backpropagation(self, y):
        n = max(y.shape)
        for k in range(self.l-1, -1, -1):
            if k == self.l-1:
                self.layers[k].error = self.calculate_output_error(y, self.cost, self.layers[self.l - 1])
            else:
                self.layers[k].error = np.multiply(self.layers[k + 1].back_error, self.layers[k].derivative_activation(self.layers[k].z))
            self.layers[k].db = np.sum(self.layers[k].error, axis=0)*1/n
            self.layers[k].dw = np.dot(self.layers[k].error.T, self.layers[k].h)*1/n
            self.layers[k].back_error = np.dot(self.layers[k].error, self.layers[k].w)

    def to_string(self):
        for i in range(self.l):
            self.layers[i].to_string()
            if i < self.l-1:
                print('         |')
                print('         |')
                print('         |')
                print('         V')

    def plt_erro(self):
        plt.scatter(np.arange(self.n_iteractions) + 1, self.errors)
        plt.show()


relu = RELU(4)
relu2 = RELU(4)
sigmoid = Sigmoid(1)
mlp = MLP([relu, sigmoid])
y = np.array([[0], [1], [1], [0]])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data = load_iris()
X = data.data[:100]
y = data.target[:100]
mlp.fit(X, y)
print(mlp.score)