from estimators.neuralnetworks.layers.Layers import Layer
import  numpy as np
class RELU(Layer):
    def g(self, z):
        z[z < 0] = 0
        return z



    def get_da(self, y):
        return None

    def derivative_activation(self,z):
        y = np.copy(z)
        y[y >= 0] = 1
        y[y < 0] = 0
        return y

    def get_name(self):
        return 'RELU'


