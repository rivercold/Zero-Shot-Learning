__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat
import numpy
import pickle
import theano

num_class = 200
word_dim = 300


def remove_unseen_in_train(Y_train, T_matrix, S_tensor, unseen_file):
    unseen_classes = numpy.loadtxt(unseen_file) - 1
    Y_train = numpy.delete(Y_train, unseen_classes, axis=1)
    T_train = numpy.delete(T_matrix, unseen_classes, axis=0)
    S_train = numpy.delete(S_tensor, unseen_classes, axis=0)

    return Y_train, T_train, S_train


def remove_seen_in_test(Y_unseen, T_matrix, S_tensor, unseen_file):
    unseen_classes = numpy.loadtxt(unseen_file) - 1
    unseen_classes = unseen_classes.astype(numpy.int)
    Y_unseen = Y_unseen[:, unseen_classes]
    T_unseen = T_matrix[unseen_classes, :]
    S_unseen = S_tensor[unseen_classes, :, :]

    return Y_unseen, T_unseen, S_unseen


def prepare_data(matroot, split_file, unseen_file, wiki_npy=None, boa_npy=None,
                 summary_npy=None, step=30, split_T_unseen=False):
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

    T_matrix = numpy.zeros((num_class, 0), dtype=theano.config.floatX)
    if wiki_npy is not None:
        T1 = prepare_wiki_data(wiki_npy)
        T_matrix = numpy.concatenate((T_matrix, T1), axis=1)

    if boa_npy is not None:
        T2 = prepare_attribute_data(boa_npy)
        T_matrix = numpy.concatenate((T_matrix, T2), axis=1)

    S_tensor = numpy.zeros((num_class, step, word_dim), dtype=theano.config.floatX)
    if summary_npy is not None:
        S_tensor = prepare_summary_data(summary_npy, step=step)

    T_seen = numpy.copy(T_matrix)
    S_seen = numpy.copy(S_tensor)

    Y_train, T_train, S_train = remove_unseen_in_train(Y_train, T_matrix, S_tensor, unseen_file)
    if split_T_unseen:  # For unseen images, only focus on unseen classes
        Y_unseen, T_unseen, S_unseen = remove_seen_in_test(Y_unseen, T_matrix, S_tensor, unseen_file)
    else:
        T_unseen = numpy.copy(T_matrix)
        S_unseen = numpy.copy(S_tensor)

    return X_train, Y_train, T_train, S_train, X_seen, Y_seen, T_seen, S_seen, X_unseen, Y_unseen, T_unseen, S_unseen


def prepare_wiki_data(npy_file):
    wiki = numpy.load(npy_file)
    return numpy.asarray(wiki, dtype=theano.config.floatX)


def prepare_attribute_data(npy_file):
    boa = numpy.load(npy_file)
    return numpy.asarray(boa, dtype=theano.config.floatX)


def prepare_summary_data(npy_file, step=30):
    summary = numpy.load(npy_file)
    summary = summary[:, :step, :]
    return numpy.asarray(summary, dtype=theano.config.floatX)
