__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat
import numpy
import pickle
import theano

num_class = 200


def prepare_vision_data_old(matroot, split_file, zsl=False, no_train=False):
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


def remove_unseen_in_train(Y_train, T_matrix, unseen_file):
    unseen_classes = numpy.loadtxt(unseen_file) - 1
    Y_train = numpy.delete(Y_train, unseen_classes, axis=1)
    T_train = numpy.delete(T_matrix, unseen_classes, axis=0)

    return Y_train, T_train


def remove_seen_in_test(Y_unseen, T_matrix, unseen_file):
    unseen_classes = numpy.loadtxt(unseen_file) - 1
    Y_unseen = Y_unseen[:, unseen_classes]
    T_unseen = T_matrix[unseen_classes, :]

    return Y_unseen, T_unseen


def prepare_data(matroot, split_file, unseen_file, wiki_npy):
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

    # unseen_classes = numpy.loadtxt(unseen_file)
    # Y_train = numpy.delete(Y_train, unseen_classes, axis=1)
    indices = numpy.arange(X_train.shape[0])
    numpy.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    X_seen, Y_seen = X[sp == 0], Y[sp == 0]
    X_unseen, Y_unseen = X[sp == -1], Y[sp == -1]

    T_matrix = prepare_wiki_data(wiki_npy)
    T_seen = numpy.copy(T_matrix)

    Y_train, T_train = remove_unseen_in_train(Y_train, T_matrix, unseen_file)
    Y_unseen, T_unseen = remove_seen_in_test(Y_unseen, T_matrix, unseen_file)

    return X_train, Y_train, T_train, X_seen, Y_seen, T_seen, X_unseen, Y_unseen, T_unseen


def prepare_wiki_data(npy_file):
    wiki = pickle.load(open(npy_file, 'rb'))
    return numpy.asarray(wiki.todense(), dtype=theano.config.floatX)
