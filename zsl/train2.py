__author__ = 'yuhongliang324'

import theano
import theano.tensor as T
from fc import FC
import timeit
import time
import load_data
import numpy
import os
import pickle
from evaluate import get_metrics
from preprocess import split_train

log_root = '../log/'
model_root = '../models/'


def validate(test_model, writer, Y_test, batch_size=100, seen='seen'):
    num_samples = Y_test.shape[0]
    batch_index = 0

    cost_total, acc_total, loss_total = 0., 0., 0.

    predictions = None

    while True:
        start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, num_samples)
        batch_index += 1

        pred, cost, loss, acc, sim = test_model(start, end, 0)

        if predictions is None:
            predictions = sim
        else:
            predictions = numpy.concatenate((predictions, sim), axis=0)

        cost_total += cost * (end - start)
        acc_total += acc * (end - start)
        loss_total += loss * (end - start)

        if end >= num_samples - 1:
            break

    cost_total /= num_samples
    acc_total /= num_samples
    loss_total /= num_samples

    labels = numpy.argmax(Y_test, axis=1)

    roc_auc, pr_auc, top1_accu, top5_accu = get_metrics(predictions, labels)
    print '\t' + seen + '\tROC-AUC = %.4f\tPR-AUC = %.4f\tTop-1 Acc = %.4f\tTop-5 Acc = %.4f'\
                        % (roc_auc, pr_auc, top1_accu, top5_accu)
    writer.write('\t' + seen + '\tROC-AUC = %.4f\tPR-AUC = %.4f\tTop-1 Acc = %.4f\tTop-5 Acc = %.4f\n'
                 % (roc_auc, pr_auc, top1_accu, top5_accu))

    return roc_auc, pr_auc, top1_accu, top5_accu


def remove_unseen_in_train(Y_train, T_matrix, unseen_file):
    unseen_classes = numpy.loadtxt(unseen_file)
    Y_train = numpy.delete(Y_train, unseen_classes, axis=1)
    T_train = numpy.delete(T_matrix, unseen_classes, axis=0)

    return Y_train, T_train


def train(V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test, obj='BCE',
          batch_size=200, max_epoch=100, store=False):

    if not obj == 'BCE':  # 0-1 coding
        Y_train = 2. * Y_train - 1.
        Y_seen = 2. * Y_seen - 1.
        Y_unseen = 2. * Y_unseen - 1.

    mlp_t_layers, mlp_v_layers = [T_train.shape[1], 300, 50], [V_train.shape[1], 200, 50]
    # mlp_t_layers, mlp_v_layers = [T_matrix.shape[1], 50], [V_train.shape[1], 200, 50]
    model = FC(mlp_t_layers, mlp_v_layers)
    symbols = model.define_functions(obj=obj)

    model_fn = model.name + '_' + obj\
               + '_tmlp_' + '-'.join([str(x) for x in mlp_t_layers])\
               + '_vmlp_' + '-'.join([str(x) for x in mlp_v_layers])\
               + '_bs_' + str(batch_size) + '_' + time.strftime("%m%d-%H-%M-%S", time.localtime())
    log_file = os.path.join(log_root, model_fn + '.log')
    writer = open(log_file, 'w')

    V_batch, T_batch, Y_batch, updates = symbols['V_batch'], symbols['T_batch'], symbols['Y_batch'], symbols['updates']
    is_train, cost, loss, acc, pred, sim = symbols['is_train'], symbols['cost'], symbols['loss'], symbols['acc'],\
                                           symbols['pred'], symbols['sim']

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    V_train_shared = theano.shared(V_train, borrow=True)
    Y_train_shared = theano.shared(Y_train, borrow=True)
    T_train_shared = theano.shared(T_train, borrow=True)

    V_seen_shared = theano.shared(V_seen, borrow=True)
    Y_seen_shared = theano.shared(Y_seen, borrow=True)

    V_unseen_shared = theano.shared(V_unseen, borrow=True)
    Y_unseen_shared = theano.shared(Y_unseen, borrow=True)

    T_test_shared = theano.shared(T_test, borrow=True)

    print 'Compiling functions ... '
    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[pred, cost, loss, acc], updates=updates,
                                  givens={
                                      V_batch: V_train_shared[start_symbol: end_symbol],
                                      Y_batch: Y_train_shared[start_symbol: end_symbol],
                                      T_batch: T_train_shared
                                  },
                                  on_unused_input='ignore')

    test_seen = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[pred, cost, loss, acc, sim],
                                  givens={
                                      V_batch: V_seen_shared[start_symbol: end_symbol],
                                      Y_batch: Y_seen_shared[start_symbol: end_symbol],
                                      T_batch: T_test_shared
                                  },
                                  on_unused_input='ignore')

    test_unseen = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[pred, cost, loss, acc, sim],
                                  givens={
                                      V_batch: V_unseen_shared[start_symbol: end_symbol],
                                      Y_batch: Y_unseen_shared[start_symbol: end_symbol],
                                      T_batch: T_test_shared
                                  },
                                  on_unused_input='ignore')

    print 'Compilation done'

    num_samples = V_train.shape[0]

    best_acc = 0.25

    for epoch_index in xrange(max_epoch):
        print 'Epoch = %d' % (epoch_index + 1)
        writer.write('Epoch = %d\n' % (epoch_index + 1))

        start_time = timeit.default_timer()

        batch_index = 0
        acc_epoch, cost_epoch, loss_epoch = 0., 0., 0.
        while True:
            start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, num_samples)
            batch_index += 1

            pred, cost, loss, acc = train_model(start, end, 1)
            cost_epoch += cost * (end - start)
            acc_epoch += acc * (end - start)
            loss_epoch += loss * (end - start)

            if end >= num_samples - 1:
                break

        cost_epoch /= num_samples
        acc_epoch /= num_samples
        loss_epoch /= num_samples

        writer.write('\tTraining\tAccuracy = %.4f\tCost = %f\tLoss = %f\n'
                     % (acc_epoch, cost_epoch, loss_epoch))

        print '\tTraining\tAccuracy = %.4f\tCost = %f\tLoss = %f'\
              % (acc_epoch, cost_epoch, loss_epoch)

        end_time = timeit.default_timer()
        print 'Train %.3f seconds for this epoch' % (end_time - start_time)
        roc_auc_seen, ap_seen, top1_accu_seen, top5_accu_seen\
            = validate(test_seen, writer, Y_seen)
        roc_auc_unseen, ap_unseen, top1_accu_unseen, top5_accu_unseen\
            = validate(test_unseen, writer, Y_unseen, seen='unseen')

        n_seen = Y_seen.shape[0]
        n_unseen = Y_unseen.shape[0]

        roc_auc = (n_seen * roc_auc_seen + n_unseen * roc_auc_unseen) / (n_seen + n_unseen)
        pr_auc = (n_seen * ap_seen + n_unseen * ap_unseen) / (n_seen + n_unseen)

        top1_accu = (n_seen * top1_accu_seen + n_unseen * top1_accu_unseen) / (n_seen + n_unseen)
        top5_accu = (n_seen * top5_accu_seen + n_unseen * top5_accu_unseen) / (n_seen + n_unseen)

        writer.write('\tMean\tROC-AUC = %.4f\tPR-AUC = %.4f\tTop-1 Acc = %.4f\tTop-5 Acc = %.4f\n'
                 % (roc_auc, pr_auc, top1_accu, top5_accu))
        print '\tMean\tROC-AUC = %.4f\tPR-AUC = %.4f\tTop-1 Acc = %.4f\tTop-5 Acc = %.4f'\
              % (roc_auc, pr_auc, top1_accu, top5_accu)

        if store and top1_accu > best_acc:
            best_acc = top1_accu
            with open(os.path.join(model_root, model_fn
                    + '_epoch_' + str(epoch_index + 1) + '_acc_' + str(top1_accu) + '.pkl'), 'wb') as pickle_file:
                pickle.dump(model, pickle_file)
        print

    writer.close()


def test1():
    dataset = 'bird-2010'
    matroot = '../features/' + dataset + '/resnet'
    split_file = '../features/' + dataset + '/zsl_split.txt'
    npy_file = '../features/wiki/wiki_features'
    V_train, Y_train, V_test, Y_test = load_data.prepare_vision_data(matroot, split_file, zsl=True)
    T_matrix = load_data.prepare_wiki_data(npy_file)
    unseen_file = '../features/' + dataset + '/unseen_classes.txt'
    Y_train, T_train = remove_unseen_in_train(Y_train, T_matrix, unseen_file)
    V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test = split_train(V_train, T_train, Y_train)
    print V_train.shape, Y_train.shape, T_train.shape, V_seen.shape, Y_seen.shape, V_unseen.shape, Y_unseen.shape, T_test.shape
    train(V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test)


'''
def test2():
    dataset = 'bird-2011'
    matroot = '../features/' + dataset + '/resnet'
    split_file = '../features/' + dataset + '/zsl_split.txt'
    npy_file = '../features/wiki/wiki_features'
    V_train, Y_train, V_seen, Y_seen, V_unseen, Y_unseen =\
        load_data.prepare_vision_data(matroot, split_file, zsl=True, split_unseen=True)
    T_matrix = load_data.prepare_wiki_data(npy_file)
    unseen_file = '../features/' + dataset + '/unseen_classes.txt'
    Y_train, T_train = remove_unseen_in_train(Y_train, T_matrix, unseen_file)
    # V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test = split_train(V_train, T_train, Y_train)
    T_test = T_matrix
    print V_train.shape, Y_train.shape, T_train.shape, V_seen.shape, Y_seen.shape, V_unseen.shape, Y_unseen.shape, T_test.shape
    train(V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test)'''


def test3():
    dataset = 'bird-2010'
    matroot = '../features/' + dataset + '/resnet'
    split_file = '../features/' + dataset + '/zsl_split.txt'
    npy_file = '../features/wiki/wiki_features'
    V, Y = load_data.prepare_vision_data(matroot, split_file, zsl=True, no_train=True)
    T_matrix = load_data.prepare_wiki_data(npy_file)
    V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test = split_train(V, T_matrix, Y, num_unseen=40)
    start_time = timeit.default_timer()
    train(V_train, Y_train, T_train, V_seen, Y_seen, V_unseen, Y_unseen, T_test, obj='BCE')
    end_time = timeit.default_timer()
    print 'Test %.3f seconds' % (end_time - start_time)


if __name__ == '__main__':
    test3()


