""" Boilerplate code for use in theano nn layers """
import numpy as np
import theano


def init_weight(shape, name='W', func=None, rng=None):
    """ Initialize a theano weights variable
    Attributes:
        shape: (M, N) - shape of the matrix
        name: Name to pass to theano.shared var constructor
        func: Custom function to apply to the values
        rng: numpy random state generator
    """
    if rng is None:
        rng = np.random

    value = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (shape[0] + shape[1])),
            high=np.sqrt(6. / (shape[0] + shape[1])),
            size=shape),
        dtype=theano.config.floatX)

    if func:
        value = func(value)

    return theano.shared(value=value, name=name, borrow=True)


def init_bias(n_dim, name='b'):
    """ Initialize a theano bias variable
    Attributes:
        n_dim: dimensionality of bias vector
        name: Name to pass to theano shared var constructor
    """
    value = np.zeros((n_dim,), dtype=theano.config.floatX)
    return theano.shared(value=value, name=name, borrow=True)
