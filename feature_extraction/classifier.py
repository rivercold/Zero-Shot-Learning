__author__ = 'yuhongliang324'

from scipy.io import loadmat, savemat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import os
import numpy

num_class = 200


def vgg_preprocess(vgg_root, feature_root):

    files = os.listdir(vgg_root)
    files.sort()
    for fn in files:
        dirpath = os.path.join(vgg_root, fn)
        if not os.path.isdir(dirpath):
            continue
        features = None
        mats = os.listdir(dirpath)
        for mat in mats:
            matpath = os.path.join(dirpath, mat)
            if not mat.endswith('.mat'):
                continue
            data = loadmat(matpath)
            data = data['feats']
            fc7 = data[0][0][1]
            fc7 = numpy.mean(fc7, axis=1)
            fc7 = numpy.reshape(fc7, (1, 4096))
            if features is None:
                features = fc7
            else:
                features = numpy.concatenate((features, fc7), axis=0)

        mdict = {'fc7': features}

        savemat(os.path.join(feature_root, fn) + '.mat', mdict)


def resnet_preprocess(resnet_root, feature_root):
    files = os.listdir(resnet_root)
    files.sort()
    for fn in files:
        dirpath = os.path.join(resnet_root, fn)
        if not os.path.isdir(dirpath):
            continue
        features = None
        mats = os.listdir(dirpath)
        for mat in mats:
            matpath = os.path.join(dirpath, mat)
            if not mat.endswith('.mat'):
                continue
            data = loadmat(matpath)
            data = data['feats']
            fc1000 = data[0][0][0]
            fc1000 = numpy.mean(fc1000, axis=1)
            fc1000 = numpy.reshape(fc1000, (1, 1000))
            if features is None:
                features = fc1000
            else:
                features = numpy.concatenate((features, fc1000), axis=0)

        mdict = {'fc7': features}

        savemat(os.path.join(feature_root, fn) + '.mat', mdict)


def prepare_data(matroot, split_file):
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
    X_train, Y_train = X[sp == 1], Y[sp == 1]
    X_test, Y_test = X[sp == 0], Y[sp == 0]
    indices = numpy.arange(X_train.shape[0])
    numpy.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    return X_train, Y_train, X_test, Y_test


def test(model, X_test, Y_test, batch_size=16):
    batch_num = X_test.shape[0] // batch_size + 1
    right = 0
    total = 0
    for batch_index in xrange(batch_num):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, X_test.shape[0] - 1)
        X_batch = X_test[start: end + 1]
        Y_batch = Y_test[start: end + 1]
        pred = model.predict_on_batch(X_batch)
        pred = numpy.argmax(pred, axis=1)
        y = numpy.argmax(Y_batch, axis=1)

        right_batch = pred[pred == y].shape[0]
        right += right_batch
        total += end - start + 1
    print 'Accuracy = %.4f' % (float(right) / total)


def optimize(X_train, Y_train, X_test, Y_test, batch_size=16):
    model = Sequential()
    model.add(Dense(num_class, input_dim=X_train.shape[1], init='glorot_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    batch_num = X_train.shape[0] // batch_size + 1

    for epoch_index in range(100):
        loss = 0.
        total = 0
        for batch_index in xrange(batch_num):
            start = batch_index * batch_size
            end = min((batch_index + 1) * batch_size, X_train.shape[0] - 1)
            X_batch = X_train[start: end + 1]
            Y_batch = Y_train[start: end + 1]
            loss += model.train_on_batch(X_batch, Y_batch)
            total += end - start + 1
        loss /= total
        print 'Epoch = %d' % (epoch_index + 1)
        test(model, X_test, Y_test)


def test1():
    matroot = '/usr0/home/hongliay/code/Zero-Shot-Learning/features/vgg'
    split_file = '/usr0/home/hongliay/zsl/data/CUB_200_2011/train_test_split.txt'
    X_train, Y_train, X_test, Y_test = prepare_data(matroot, split_file)
    optimize(X_train, Y_train, X_test, Y_test)


def test2():
    resnet_root = '/usr0/home/hongliay/code/Zero-Shot-Learning/features/resnet'
    feature_root = '/usr0/home/hongliay/code/Zero-Shot-Learning/features/resnet_ppd'
    resnet_preprocess(resnet_root, feature_root)

def test3():
    matroot = '/usr0/home/hongliay/code/Zero-Shot-Learning/features/resnet'
    split_file = '/usr0/home/hongliay/zsl/data/CUB_200_2011/train_test_split.txt'
    X_train, Y_train, X_test, Y_test = prepare_data(matroot, split_file)
    optimize(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    test2()
