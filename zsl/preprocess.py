__author__ = 'yuhongliang324'

from load_data import *
import random
import cPickle as pickle
import numpy as np
#import pickle

# return: split result:
# 1 - training, 0 - seen test, -1 - unseen test
def split(matroot, split_file, unseen_file):
    Y = None
    files = os.listdir(matroot)
    files.sort()
    for fn in files:
        if not fn.endswith('.mat'):
            continue
        data = loadmat(os.path.join(matroot, fn))
        features = data['fc7']
        label = int(fn[:3]) - 1
        y = label * numpy.ones((features.shape[0],))
        if Y is None:
            Y = y
        else:
            Y = numpy.concatenate((Y, y))

    sp = numpy.ones_like(Y)

    remain = []

    unseen_classes = random.sample(range(num_class), 40)
    unseen_classes.sort()
    for i in xrange(Y.shape[0]):
        if Y[i] in unseen_classes:
            sp[i] = -1
        else:
            remain.append(i)
    test_index = random.sample(remain, int(len(remain) * 0.2))
    for ti in test_index:
        sp[ti] = 0

    writer = open(split_file, 'w')
    for i in xrange(sp.shape[0]):
        writer.write(str(int(sp[i])) + '\n')
    writer.close()

    writer = open(unseen_file, 'w')
    for tc in unseen_classes:
        writer.write(str(tc) + '\n')
    writer.close()


# split for validation
def split_train(V_train, T_train, Y_train, num_unseen=30):

    y_vec = numpy.argmax(V_train, axis=1)

    sp = numpy.ones((V_train.shape[0],))

    remain = []

    unseen_classes = random.sample(range(Y_train.shape[1]), num_unseen)
    unseen_classes.sort()
    for i in xrange(Y_train.shape[0]):
        if y_vec[i] in unseen_classes:
            sp[i] = -1
        else:
            remain.append(i)
    test_index = random.sample(remain, int(len(remain) * 0.2))
    for ti in test_index:
        sp[ti] = 0

    V_seen, V_unseen = V_train[sp == 0], V_train[sp == -1]
    V_train = V_train[sp == 1]

    T_test = numpy.copy(T_train)
    T_train = numpy.delete(T_train, unseen_classes, axis=0)

    Y_seen, Y_unseen = Y_train[sp == 0], Y_train[sp == -1]
    Y_train = numpy.delete(Y_train[sp == 1], unseen_classes, axis=1)

    return V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test


def load_feature_map(pkl_path):
    from fc import FC
    model = pickle.load(open(pkl_path))
    return model.W_t_mlp[0].get_value()


def test1():
    split('../features/bird-2010/resnet', '../features/bird-2010/zsl_split.txt',
          '../features/bird-2010/unseen_classes.txt')


def test2():
    W = load_feature_map('../models/'
                     'fc_BCE_tmlp_6366-50_vmlp_1000-200-50_bs_200_1108-19-04-35_epoch_85_acc_0.262886597938.pkl')
    with open("../features/wiki/vocabulary","r") as outfile:
        vocabulary = pickle.load(outfile)
    vocabulary = convert_vocabulary(vocabulary)
    for i in range(50):
        t = W.T[i]
        index = np.argsort(-t)[:10]
        print index
        for id in index:
            print vocabulary[id],
        print "\n"
    print len(vocabulary)


def convert_vocabulary(vocabulary):
    new_vocab = {}
    for word, id in vocabulary.iteritems():
        new_vocab[id] = word
    return new_vocab

if __name__ == '__main__':
    test2()


    #file_path = '../models/fc_BCE_tmlp_6366-50_vmlp_1000-200-50_bs_200_1108-19-04-35_epoch_85_acc_0.262886597938.pkl'
    #with open(file_path) as outfile:
    #    p = pickle.load(outfile)

