""" Training helpers """
import timeit

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report, f1_score
import theano
from theano import tensor as T

from softmax import SoftmaxLayer, MLP
from autoencoders import StackedDenoisingAutoencoder


def train_sgd_mnist(clf, alpha=0.01, n_epochs=1000, batch_size=20, **kwargs):
    """ Run a training session for the MNIST dataset using a given classifier

    Attributes:
        clf: Classifier. Must be uninitialized, or else X should be given
        alpha: learning_rate
        n_epochs: number of epochs
        batch_size: number of samples for each gradient update
        kwargs: Arguments to pass to clf.__init__
    """
    # Data loading
    mnist = datasets.fetch_mldata("MNIST original")
    X_all, y_all = mnist.data / 255, mnist.target.astype(np.int32)

    X_train = theano.shared(value=X_all[:60000].astype(theano.config.floatX))
    X_test = X_all[60000:].astype(np.float32)
    y_train = theano.shared(value=y_all[:60000])
    y_test = y_all[60000:]

    n_batches = y_train.get_value(borrow=True).shape[0] // batch_size

    # Placeholders
    index = T.lscalar()
    if hasattr(clf, 'X'):
        X = clf.X
    else:
        X = T.matrix('X')
        clf = clf(X=X, **kwargs)

    y = T.ivector('y')

    cost = clf.cost(y)

    gparams = [T.grad(cost, p) for p in clf.params]
    updates = [(p, p - alpha * gp) for p, gp in zip(clf.params, gparams)]

    train_function = theano.function(
        inputs=[index], outputs=cost, updates=updates,
        givens={X: X_train[index * batch_size:(index+1) * batch_size],
                y: y_train[index * batch_size:(index+1) * batch_size]})

    predict_function = theano.function(inputs=[X], outputs=clf.y_pred)

    start_time = timeit.default_timer()

    epoch = 0

    while (epoch < n_epochs):
        epoch += 1
        epoch_start = timeit.default_timer()
        epoch_costs = []
        for minibatch_index in range(n_batches):
            epoch_costs.append(train_function(minibatch_index))

        this_loss = np.mean(epoch_costs)
        print("\rEpoch: %d, loss: %.2f, duration: %.1fs" %
            (epoch+1, this_loss,
             timeit.default_timer() - epoch_start), end='')

        if (epoch + 1) % 25 == 0:
            print("\n\t Test set f1_score: %.2f" %
                f1_score(y_test, predict_function(X_test), average='macro'))

    duration = (timeit.default_timer() - start_time) / 60
    print('')
    print(classification_report(y_test, predict_function(X_test)))
    print("The code run for %d epochs, ~ %.0f epochs/min, total time %.2fm" %
        (epoch, epoch / duration, duration))


def pretrain(clf, alpha, n_epochs, batch_size):
    """ Pretraining for stacked denoising autoencoders
    OBJECT MUST BE INITIALIZED!"""
    X = clf.X

    mnist = datasets.fetch_mldata("MNIST original")
    X_all = theano.shared(value=(mnist.data / 255).astype(theano.config.floatX))

    n_batches = X_all.get_value(borrow=True).shape[0] // batch_size

    fns = clf.pretraining_fns(X_all, batch_size, alpha)

    start_time = timeit.default_timer()
    for i, func in enumerate(fns):
        for epoch in range(n_epochs):
            c = []
            epoch_start = timeit.default_timer()
            for batch_index in range(n_batches):
                c.append(func(index=batch_index))
            print("\rPretraining layer %d, epoch: %d, loss: %.2f, "
                  "duration: %.1fs" %
                  (i, (epoch+1), np.mean(c, dtype=float),
                   timeit.default_timer() - epoch_start), end='')
        print('')

    duration = (timeit.default_timer() - start_time) / 60
    print("Finished pretraining in %.1fmin" % duration)


if __name__ == "__main__":
    """
    print("==============\n"
          "SIMPLE SOFTMAX\n"
          "==============")
    train_sgd_mnist(SoftmaxLayer,
                    alpha=0.01,
                    n_epochs=150,
                    batch_size=200,
                    shape=(28*28, 10))

    print("===\n"
          "MLP\n"
          "===")
    train_sgd_mnist(MLP,
                    alpha=0.01,
                    n_epochs=150,
                    batch_size=200,
                    shape=(28*28, 400, 10),
                    L1_reg=0.,
                    L2_reg=0.0001)

    """
    print("====================================\n"
          "DENOISING AUTOENCODER w/ PRETRAINING\n"
          "====================================")

    X = T.matrix('X')
    clf = StackedDenoisingAutoencoder(X, [28*28, 400, 400, 400, 10],
                                         [.1, .2, .3])

    pretrain(clf,
             alpha=0.001,
             n_epochs=20,
             batch_size=1)

    train_sgd_mnist(clf,
                    alpha=0.1,
                    n_epochs=100,
                    batch_size=500,
                    X=X)
