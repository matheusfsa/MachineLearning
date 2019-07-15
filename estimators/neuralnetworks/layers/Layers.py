import  numpy as np


class Layer:
    def __init__(self , units):
        self.z = np.array([])
        self.w = np.array([])
        self.b = np.array([])
        self.A = np.array([])
        self.dw = None
        self.h = None
        self.db = None
        self.back_error = None
        self.error = None
        self.units = units
        self.input_dim = 0

    def initialize_parameters(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.rand(self.units, input_dim)
        self.b = np.random.rand(self.units)

    def g(self, z):
        raise NotImplementedError

    def restart(self):
        self.z = np.array([])
        self.A = np.array([])
        self.dw = None
        self.h = None
        self.db = None
        self.back_error = None
        self.error = None

    def get_da(self, y):
        raise NotImplementedError

    def activation(self, X):
        self.h = X
        self.z = np.dot(X, self.w.T) + self.b
        self.A = self.g(self.z)
        return self.A

    def derivative_activation(self, z):
        raise NotImplementedError

    def foward(self, A):
        self.A = self.activation(A)
        return A



    def get_derivative(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def update(self, learning_ratio, n):
        self.b = (1 - learning_ratio) * self.b + learning_ratio * self.db
        self.w = (1 - learning_ratio) * self.w + learning_ratio * self.dw

    def to_string(self):
        print('-'*11)
        print('Camada:', self.get_name())
        print('Unidades:', self.units)
        print('Pesos:\n',  self.w)
        print('dW:\n', self.dw)
        print('Bias: \n', self.b)
        print('db:\n', self.db)
        print('-' * 11)

