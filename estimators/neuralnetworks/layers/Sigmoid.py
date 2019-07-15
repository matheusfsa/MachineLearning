from estimators.neuralnetworks.layers.Layers import Layer
import  numpy as np


class Sigmoid(Layer):

    def g(self, z):
        return 1 / (1 + np.exp(-z))


    def optimize(self):
        return None

    def derivative_activation(self, z):
        sig = lambda x: np.exp(x) / (1 + np.exp(x))
        return sig(z) * (1 - sig(z))

    def get_da(self, y):
        a = self.g(self.z)
        return a*(1 - a)

    def get_name(self):
        return 'Sigmoid'


