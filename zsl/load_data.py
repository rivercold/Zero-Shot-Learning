__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat
import numpy
import pickle
import theano

num_class = 200


def prepare_vision_data(matroot, split_file):
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
    sp = sp[:, 1]
    Y = numpy.asarray(Y, dtype=theano.config.floatX)
    X_train, Y_train = X[sp == 1], Y[sp == 1]
    X_test, Y_test = X[sp == 0], Y[sp == 0]
    indices = numpy.arange(X_train.shape[0])
    numpy.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    return X_train, Y_train, X_test, Y_test


def prepare_wiki_data(npy_file):
    wiki = pickle.load(open(npy_file, 'rb'))
    return numpy.asarray(wiki.todense(), dtype=theano.config.floatX)

