""" Simple denoising autoencoders demonstration using theano
    from: http://deeplearning.net/tutorial/dA.html
"""
# pylint: disable=too-many-arguments, invalid-name
# These are common in Neural Network applications
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from softmax import SoftmaxLayer, HiddenLayer
from helpers import init_weight, init_bias


class DenoisingAutoencoder:
    """ Denoising Autoencoder layer.

    Attributes:
        X: input (theano placeholder)
        shape: (n_in, n_hidden)
        rng: np.RandomState instance
        theano_rng: leave empty, gets populated from rng
        W: custom weight
        b_in: custom biases for input layer
        b_hidden: custom biases for hidden layer
    """
    def __init__(self, X, shape, rng=None, theano_rng=None,
                 W=None, b_in=None, b_hidden=None):

        if rng is None:
            rng = np.random

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        if W is None:
            W = init_weight(shape)

        if b_in is None:
            b_in = init_bias(shape[-1])

        if b_hidden is None:
            b_hidden = init_bias(shape[0])

        self.X = X
        self.W = W
        self.b_in = b_in
        self.b_hidden = b_hidden
        self.theano_rng = theano_rng

        self.params = [self.W, self.b_in, self.b_hidden]

    def get_hidden_values(self, X):
        """ Compute the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(X, self.W) + self.b_in)

    def get_reconstructed_input(self, hidden):
        """ Reconstruct the input given hidden layer activation """
        return T.nnet.sigmoid(T.dot(hidden, self.W.T) + self.b_hidden)

    def get_dropout_input(self, X, dropout):
        """ Drop randomly selected signals for robustness (dropout)
        Notice we divide the resulting vector by dropout = inverse dropout
        so we don't have to do this during model predictions
        """
        mask = self.theano_rng.binomial(size=X.shape, n=1, p=1-dropout,
                                        dtype=theano.config.floatX)
        inverse = mask / dropout
        return inverse * X

    def get_cost_updates(self, dropout, alpha):
        """Compute the cost and apply a training step """
        tilde_X = self.get_dropout_input(self.X, dropout)
        y = self.get_hidden_values(tilde_X)
        z = self.get_reconstructed_input(y)

        loss = -T.sum(self.X * T.log(z) + (1 - self.X) * T.log(1 - z), axis=1)
        cost = T.mean(loss)

        gparams = T.grad(cost, self.params)

        updates = [(p, p - alpha * gp)
                   for p, gp in zip(self.params, gparams)]

        return (cost, updates)

    def reconstruct(self):
        """ Propagate through network and reconstruct input X """
        X_enc = self.get_hidden_values(self.X)
        return self.get_reconstructed_input(X_enc)


class StackedDenoisingAutoencoder:
    """ Autoencoder + MLP Combo

    Attributes:
        X: input (theano placeholder)
        shape: (n_in, n_hidden1, n_hidden2 etc, n_out)
        dropouts: list with dropout parameters for each hidden layer
        rng: np.RandomState instance
        theano_rng: leave empty, gets populated from rng
    """
    def __init__(self, X, shape, dropouts, rng=None, theano_rng=None):
        self.sigmoid_layers = []
        self.autoencoding_layers = []
        self.params = []

        if rng is None:
            rng = np.random

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.X = X
        self.dropouts = dropouts

        for i in range(0, len(shape)-2):
            X_layer = self.X if i == 0 else self.sigmoid_layers[-1].output
            sigmoid_layer = HiddenLayer(X_layer, shape[i:i+2], rng=rng,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            autoencoding_layer = DenoisingAutoencoder(
                X_layer, shape[i:i+2], rng=rng, theano_rng=theano_rng,
                W=sigmoid_layer.W, b_in=sigmoid_layer.b)

            self.autoencoding_layers.append(autoencoding_layer)

        self.softmax_layer = SoftmaxLayer(self.sigmoid_layers[-1].output,
                                          shape[-2:])
        self.y_pred = self.softmax_layer.y_pred
        self.cost = self.softmax_layer.cost
        self.errors = self.softmax_layer.errors

        self.params.extend(self.softmax_layer.params)

    def pretraining_fns(self, X_train, batch_size, alpha):
        index = T.lscalar('index')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        fns = []
        for i, layer in enumerate(self.autoencoding_layers):
            cost, updates = layer.get_cost_updates(self.dropouts[i], alpha)
            fn = theano.function(
                inputs=[index], outputs=cost, updates=updates,
                givens={self.X: X_train[batch_begin:batch_end]})
            fns.append(fn)
        return fns
