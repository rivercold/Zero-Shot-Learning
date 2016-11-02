__author__ = 'yuhongliang324'

import theano
import theano.tensor as T
from fc import FC
import timeit
import load_data
import numpy


def validate(test_model, num_samples, batch_size=100):
    start_time = timeit.default_timer()
    batch_index = 0
    acc_total, cost_total, loss_total = 0., 0., 0.
    while True:
        start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, num_samples)
        batch_index += 1

        pred, cost, loss, acc = test_model(start, end, 0)

        cost_total += cost * (end - start)
        acc_total += acc * (end - start)
        loss_total += loss * (end - start)

        if end >= num_samples - 1:
            break

    cost_total /= num_samples
    acc_total /= num_samples
    loss_total /= num_samples

    print '\tTesting\tAccuracy = %.4f\tTraining cost = %f\tTraining loss = %f'\
              % (acc_total, cost_total, loss_total)

    end_time = timeit.default_timer()
    print 'Test %.3f seconds\n' % (end_time - start_time)


def train(V_matrix, T_matrix, Y_matrix, V_matrix_test, Y_matrix_test, obj='BCE', batch_size=200, max_epoch=100):

    if not obj == 'BCE':  # 0-1 coding
        Y_matrix = 2. * Y_matrix - 1.
        Y_matrix_test = 2. * Y_matrix_test - 1.

    mlp_t_layers, mlp_v_layers = [T_matrix.shape[1], 300, 50], [V_matrix.shape[1], 200, 50]
    model = FC(mlp_t_layers, mlp_v_layers)
    symbols = model.define_functions(obj=obj)

    V_batch, T_batch, Y_batch, updates = symbols['V_batch'], symbols['T_batch'], symbols['Y_batch'], symbols['updates']
    is_train, cost, loss, acc, pred = symbols['is_train'], symbols['cost'], symbols['loss'], symbols['acc'], symbols['pred']

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    V_matrix_shared = theano.shared(V_matrix, borrow=True)
    Y_matrix_shared = theano.shared(Y_matrix, borrow=True)
    T_matrix_shared = theano.shared(T_matrix, borrow=True)

    V_matrix_test_shared = theano.shared(V_matrix_test, borrow=True)
    Y_matrix_test_shared = theano.shared(Y_matrix_test, borrow=True)

    print 'Compiling functions ... '
    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[pred, cost, loss, acc], updates=updates,
                                  givens={
                                      V_batch: V_matrix_shared[start_symbol: end_symbol],
                                      Y_batch: Y_matrix_shared[start_symbol: end_symbol],
                                      T_batch: T_matrix_shared
                                  },
                                  on_unused_input='ignore')
    test_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[pred, cost, loss, acc],
                                  givens={
                                      V_batch: V_matrix_test_shared[start_symbol: end_symbol],
                                      Y_batch: Y_matrix_test_shared[start_symbol: end_symbol],
                                      T_batch: T_matrix_shared
                                  },
                                  on_unused_input='ignore')
    print 'Compilation done'

    num_samples = V_matrix.shape[0]

    for epoch_index in xrange(max_epoch):
        print 'Epoch = %d' % (epoch_index + 1)

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

        print 'Epoch = %d\tTraining Accuracy = %.4f\tTraining cost = %f\tTraining loss = %f'\
              % (epoch_index + 1, acc_epoch, cost_epoch, loss_epoch)

        end_time = timeit.default_timer()
        print 'Train %.3f seconds for this epoch' % (end_time - start_time)

        validate(test_model, V_matrix_test.shape[0])


def test1():
    matroot = '../features/resnet'
    split_file = '../features/train_test_split.txt'
    npy_file = '../features/wiki/wiki_features'
    V_matrix_train, Y_matrix_train, V_matrix_test, Y_matrix_test = load_data.prepare_vision_data(matroot, split_file)
    T_matrix = load_data.prepare_wiki_data(npy_file)
    train(V_matrix_train, T_matrix, Y_matrix_train, V_matrix_test, Y_matrix_test)

if __name__ == '__main__':
    test1()


