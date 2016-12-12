""" Multilayer softmax classification """
# pylint: disable=invalid-name, too-many-arguments
# Variable names such as X and W are common in machine learning applications
import numpy as np
from theano import tensor as T

from helpers import init_weight, init_bias


class SoftmaxLayer:
    """ Logistic regression classifier """
    def __init__(self, X, shape):
        self.X = X
        self.W = init_weight(shape)
        self.b = init_bias(shape[-1])
        self.output = T.nnet.softmax(T.dot(self.X, self.W) + self.b)
        self.y_pred = T.argmax(self.output, axis=1)
        self.params = [self.W, self.b]

    def cost(self, y):
        """ Negative log likelihood """
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Error-rate """
        return T.mean(T.neq(self.y_pred, y))


class HiddenLayer:
    """ Dead simple hidden layer implementation """
    def __init__(self, X, shape, rng=None, W=None, b=None, activation=T.tanh):
        if rng is None:
            rng = np.random

        if W is None:
            W = init_weight(shape, func=lambda x: x*4)

        if b is None:
            b = init_bias(shape[-1])

        self.X = X
        self.W = W
        self.b = b
        self.output = activation(T.dot(self.X, self.W) + self.b)

        self.params = [self.W, self.b]


class MLP:
    """ Single-hidden-layer perceptron
    Attributes:
        X: input (theano placeholder)
        shape: (n_in, n_hidden, n_out)
        rng: np.RandomState instance
        L1_reg: L1 regularization term
        L2_reg: L2 regularization term
    """

    def __init__(self, X, shape, rng=None, L1_reg=0., L2_reg=0.):
        self.X = X

        if rng is None:
            rng = np.random

        self.L1_reg, self.L2_reg = L1_reg, L2_reg

        self.hidden_layer = HiddenLayer(X, shape[:2], rng)
        self.softmax_layer = SoftmaxLayer(self.hidden_layer.output, shape[1:])

        self.L1 = (abs(self.hidden_layer.W).sum() +
                   abs(self.softmax_layer.W).sum())

        self.L2 = ((self.hidden_layer.W ** 2).sum() +
                   (self.softmax_layer.W ** 2).sum())

        self.cost_func = self.softmax_layer.cost
        self.errors = self.softmax_layer.errors
        self.y_pred = self.softmax_layer.y_pred
        self.params = self.hidden_layer.params + self.softmax_layer.params

    def cost(self, y):
        """ Negative log likelihood with L1 and/or L2 regularization """
        return (self.cost_func(y)
                + self.L1_reg * self.L1
                + self.L2_reg * self.L2)
