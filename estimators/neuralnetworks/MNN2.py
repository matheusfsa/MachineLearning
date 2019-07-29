import numpy as np
class Neuron:
    def __init__(self, activation1='linear', activation2='linear'):
        self.q = None
        self.input_dim = 0
        self.d_activation1 = Neuron.get_derivative_activation_function(activation1)
        self.activation1 = Neuron.get_activation_function(activation1)
        self.d_activation2 = Neuron.get_derivative_activation_function(activation2)
        self.activation2 = Neuron.get_activation_function(activation2)
        self.p = np.array([])
        self.w = np.array([])
        self.v = np.array([])
        self.y_out = np.array([])
        self.y_in = np.array([])
        self.local_gradient = 0.0
        self.connected_layer = None
        self.is_output = False
        self.error_out = 0.0
        self.y_true = None
        self.dq = None

    def receive_signal(self, y):
        self.y_in = y

    def compute_local_gradiente(self):
        if self.is_output:
            self.error_out = self.y_true - self.y_out
        else:
            for i in range(len(self.connected_layer)):
                self.error_out += self.connected_layer[i].error_out
        self.local_gradient = -self.error_out * self.d_activation1(self.v)

    def send_error(self):
        self.error_out = self.local_gradient * self.w

    def compute(self, y):
        self.y_in = y
        self.p = np.dot(self.y_in, self.q)
        self.w = self.activation2(self.p)
        self.v = np.dot(self.y_in, self.w)
        self.y_out = self.activation1(self.v)

    def get_dq(self, learning_ratio):
        dq = np.zeros_like(self.q)
        for i in range(self.input_dim):
            for k in range(self.input_dim):
                dq[i, k] = -learning_ratio*self.local_gradient*self.d_activation2(self.p[i])*self.y_in[i]*self.y_in[k]
        self.dq = dq

    def update_q(self):
        self.q += self.dq

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


class Network:
    def __init__(self, layers=[[Neuron(activation1='relu'), Neuron(activation1='relu')], [Neuron(activation1='sigmoid')]], learning_ratio=0.3):
        self.layers = layers
        self.L = len(layers)
        self.learning_ratio = learning_ratio

    def init_network(self, X, y):
        m = X.shape[0]
        for i in range(self.L):
            for j in range(len(self.layers[i])):
                if i < self.L-1:
                    self.layers[i][j].connected_layer = self.layers[i + 1]
                    if i > 0:
                        self.layers[i][j].input_dim = len(self.layers[i - 1])
                    else:
                        self.layers[i][j].input_dim = m
                else:
                    self.layers[i][j].is_output = True
                    self.layers[i][j].y_true = y
                    self.layers[i][j].input_dim = len(self.layers[i - 1])
                self.layers[i][j].q = np.random.rand(self.layers[i][j].input_dim, self.layers[i][j].input_dim)*0.1

    def foward_propagation(self, X):
        signal = X
        for i in range(self.L):
            for j in range(len(self.layers[i])):
                new_signal = np.zeros(len(self.layers[i]))
                self.layers[i][j].compute(signal)
                new_signal[j] = self.layers[i][j].y_out
            signal = new_signal
        return signal

    def predict(self, X):
        y_pred = self.foward_propagation(X)
        if y_pred > 0.5:
            return 1
        else:
            return 0

    def backward_propagation(self):
        dqs = []
        for i in range(self.L-1, -1, -1):
            dq = []
            for j in range(len(self.layers[i])):
                self.layers[i][j].compute_local_gradiente()
                dq.append(self.layers[i][j].get_dq(self.learning_ratio))
            dqs.append(dq)
        return dqs

    def update_weights(self):
        for i in range(self.L-1, -1, -1):
            for j in range(len(self.layers[i])):
                self.layers[i][j].update_q()

    def fit(self, X, y):
        self.init_network(X, y)
        for i in range(2):
            y_pred = self.foward_propagation(X)
            print(y - y_pred)
            self.backward_propagation()
            self.update_weights()

n = Network()
n.fit(np.array([0, 0]), 0)
n.fit(np.array([1, 0]), 1)
n.fit(np.array([0, 1]), 1)
n.fit(np.array([1, 1]), 0)