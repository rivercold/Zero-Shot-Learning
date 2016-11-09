__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat
import numpy
import pickle
import theano

num_class = 200


def prepare_vision_data(matroot, split_file, zsl=False, no_train=False):
    X, Y = None, None
    files = os.listdir(matroot)
    files.sort()
    for fn in files:
        if not fn.endswith('.mat'):
            continue
        data = loadmat(os.path.join(matroot, fn))
        features = data['fc7']
        label = int(fn[:3]) - 1
        y = numpy.zeros((features.shape[0], num_class))
        y[:, label] = 1
        if X is None:
            X = features
            Y = y
        else:
            X = numpy.concatenate((X, features), axis=0)
            Y = numpy.concatenate((Y, y), axis=0)
    sp = numpy.loadtxt(split_file, delimiter=' ')
    if not zsl:
        sp = sp[:, 1]  # For train_test_split.txt only
    Y = numpy.asarray(Y, dtype=theano.config.floatX)

    if no_train:
        indices = numpy.arange(X.shape[0])
        numpy.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        return X, Y

    X_train, Y_train = X[sp == 1], Y[sp == 1]

    # unseen_classes = numpy.loadtxt(unseen_file)
    # Y_train = numpy.delete(Y_train, unseen_classes, axis=1)
    indices = numpy.arange(X_train.shape[0])
    numpy.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    X_test, Y_test = X[sp != 1], Y[sp != 1]

    return X_train, Y_train, X_test, Y_test


def prepare_wiki_data(npy_file):
    wiki = pickle.load(open(npy_file, 'rb'))
    return numpy.asarray(wiki.todense(), dtype=theano.config.floatX)
