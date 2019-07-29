import numpy as np

class MNN:
    def __init__(self, n_units=[2, 1], activations1=['relu', 'sigmoid'], activations2=['linear', 'linear'], learning_ratio=0.3, epochs=100):
        self.L = len(n_units)
        self.n_units = n_units
        self.layers = []
        self.activations1 = activations1
        self.activations2 = activations2
        self.set_functions()
        self.learning_ratio = learning_ratio
        self.activations = []
        self.d_activations = []
        self.set_functions()
        self.Qs = []
        self.dQs = []
        self.epochs = epochs

    @staticmethod
    def get_activation_function(name):
        if name == 'sigmoid':
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif name == 'linear':
            return lambda x: x
        elif name == 'relu':
            def relu(x):
                y = np.copy(x)
                y[y < 0] = 0
                return y
            return relu
        else:
            return lambda x: x

    @staticmethod
    def get_derivative_activation_function(name):
        if name == 'sigmoid':
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y

            return relu_diff
        else:
            return lambda x: np.ones_like(x)

    def set_functions(self):
        for i in range(self.L):
            layer = dict()
            layer['a1'] = MNN.get_activation_function(self.activations1[i])
            layer['a2'] = MNN.get_activation_function(self.activations2[i])
            layer['da1'] = MNN.get_derivative_activation_function(self.activations1[i])
            layer['da2'] = MNN.get_derivative_activation_function(self.activations2[i])
            self.layers.append(layer)

    def initialize_parameters(self, m):
        input_dim = m
        for i in range(self.L):
            units = self.n_units[i]
            self.Qs.append(np.random.rand(units,  input_dim, input_dim))
            self.dQs.append([])
            input_dim = units
    def get_p(self, q, X):
        np.dot(q, X.T).reshape(X.shape[0], X.shape[0])

    def get_a_da(self, X, l):
        layer = self.layers[l]
        q = self.Qs[l]
        p = np.transpose(np.dot(X, q), axes=(0, 2, 1))
        a2 = layer['a2'](p)
        da2 = layer['da2'](p)
        v = np.matmul(X.reshape(X.shape[0], 1, X.shape[1]), p).reshape(X.shape[0], p.shape[2])
        # v = np.sum(np.multiply(X, p), axis=2).T
        a1 = layer['a1'](v)
        da1 = layer['da1'](v)
        return a1, da1, a2, da2

    def foward_propagation(self, X):
        activations1 = [X]
        d_activations1 = [X]
        activations2 = [X]
        d_activations2 = [X]
        for i in range(1, self.L+1):
            a1, da1, a2, da2 = self.get_a_da(activations1[i-1], i-1)
            activations1.append(a1)
            d_activations1.append(da1)
            activations2.append(a2)
            d_activations2.append(da2)
        return activations1, d_activations1, activations2, d_activations2

    def predict(self,X):
        y_pred = self.foward_propagation(X)[0][self.L-1]
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

    def get_yik(self, a):
        res = np.zeros(shape=(a.shape[0], a.shape[1], a.shape[1]))
        for i in range(a.shape[0]):
            res[i] = np.outer(a[i], a[i])
        return res

    def get_la(self, a, b):
        res = np.zeros(shape=(a.shape[0], a.shape[1], a.shape[2]))
        for i in range(a.shape[0]):
            res[i] = np.multiply(a[i], b[i])
        return res

    def back_propagation(self, activations1, d_activations1, activations2, d_activations2, y):
        n = y.shape[0]
        erro = y - activations1[self.L]
        dQs = []
        for i in range(self.L, 0, -1):
            print(i)
            local_grad_l = -np.multiply(erro, d_activations1[i])
            print('erro:', erro.shape)
            print('d_activations1[i]:', d_activations1[i].shape)
            print('activations1[i-1]:', activations1[i-1].shape)
            y_ik = self.get_yik(activations1[i-1])
            print('local_grad_l:', local_grad_l.shape)
            print('activations1[i]:', activations1[i].shape)
            print('activations2[i]:', activations2[i].shape)
            print('d_activations1[i]:', d_activations1[i].shape)
            print('d_activations2[i]:', d_activations2[i].shape)
            print('y_ik:', y_ik.shape)
            print('np.multiply(local_grad_l.T, d_activations2[i]):', self.get_la(d_activations2[i], local_grad_l).shape)
            dq = self.learning_ratio/n * np.multiply(self.get_la(d_activations2[i], local_grad_l), y_ik)
            dQs.append(dq)
            erro = np.matmul(local_grad_l.T, activations2[i])
            erro = erro.reshape(erro.shape[0], erro.shape[1])
        return dQs

    def update_qs(self, dQs):
        for i in range(self.L-1, -1, -1):
            self.Qs[i] += dQs[(self.L-1)]

    def loss_function(self, name):
        if name == 'cross-entropy':
            return lambda y_true, y_pred: (-y_true*np.log(y_pred) - (1 - y_true)*np.log(1-y_pred)).mean()

    def get_error(self, X, y):
        epsilon = 1e-12
        y_pred = self.foward_propagation(X)[0][self.L-1]
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        erro = self.loss_function('cross-entropy')(y, y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        score = np.sum(y_pred == y) / y.shape[0]
        return erro, score

    def fit(self, X, y):
        X_b = np.ones((X.shape[0], X.shape[1]+1))
        X_b[:, :-1] = X
        y = y.reshape(-1, 1)
        mnn.initialize_parameters(X_b.shape[1])
        for i in range(self.epochs):
            activations1, d_activations1, activations2, d_activations2 = self.foward_propagation(X_b)
            dqs = self.back_propagation(activations1, d_activations1, activations2, d_activations2, y)
            self.update_qs(dqs)
            print(self.get_error(X_b, y))


X = np.random.rand(2, 4)
X_b = np.ones((X.shape[0], X.shape[1]+1))
X_b[:, :-1] = X
mnn = MNN()
mnn.initialize_parameters(5)
#a = mnn.get_a_da(X_b,  0)
#mnn.fit(X,np.random.rand(2,1))
activations1, d_activations1, activations2, d_activations2 = mnn.foward_propagation(X_b)
dqs = mnn.back_propagation(activations1, d_activations1, activations2, d_activations2, np.random.rand(2,1))