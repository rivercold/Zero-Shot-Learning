__author__ = 'yuhongliang324'

import theano
import theano.tensor as T
from fc import FC
import timeit
import load_data


def train(V_matrix, T_matrix, Y_matrix, obj='BCE', batch_size=100, max_epoch=100):
    mlp_t_layers, mlp_v_layers = [T_matrix.shape[1], 300, 50], [V_matrix.shape[1], 300, 50]
    model = FC(mlp_t_layers, mlp_v_layers)
    symbols = model.define_functions(obj=obj)

    V_batch, T_batch, Y, updates = symbols['V_batch'], symbols['T_batch'], symbols['Y'], symbols['updates']
    is_train, cost, acc, pred = symbols['is_train'], symbols['cost'], symbols['acc'], symbols['pred']

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    V_matrix_shared = theano.shared(V_matrix, borrow=True)
    Y_matrix_shared = theano.shared(Y_matrix, borrow=True)
    T_matrix_shared = theano.shared(T_matrix, borrow=True)

    print 'Compiling functions ... '
    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                            outputs=[pred, cost, acc], updates=updates,
                            givens={
                                V_batch: V_matrix_shared[start_symbol: end_symbol],
                                Y: Y_matrix_shared[start_symbol: end_symbol],
                                T_batch: T_matrix_shared
                            },
                            on_unused_input='ignore')
    print 'Compilation done'

    num_samples = V_matrix.shape[0]

    for epoch_index in xrange(max_epoch):
        print 'Epoch = %d' % (epoch_index + 1)

        start_time = timeit.default_timer()

        batch_index = 0
        cost_epoch, acc_epoch = 0., 0.
        while True:
            start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, num_samples)
            batch_index += 1

            pred, cost, acc = train_model(start, end, 1)
            cost_epoch += cost * (end - start)
            acc_epoch += acc * (end - start)

            print '\tAccuracy = %.4f\tTraining cost = %f' % (acc, cost)

            if end >= num_samples - 1:
                break

        cost_epoch /= num_samples
        acc_epoch /= num_samples

        print 'Epoch = %d\tAccuracy = %.4f\tTraining cost = %f' % (epoch_index + 1, acc_epoch, cost_epoch)

        end_time = timeit.default_timer()
        print 'Ran %.3f seconds for this epoch\n' % (end_time - start_time)


def test1():
    matroot = '/usr0/home/hongliay/code/Zero-Shot-Learning/features/resnet'
    split_file = '/usr0/home/hongliay/zsl/data/CUB_200_2011/train_test_split.txt'
    npy_file = '/usr0/home/hongliay/code/Zero-Shot-Learning/features/wiki/wiki_features'
    V_matrix_train, Y_matrix_train, V_matrix_test, V_matrix_test = load_data.prepare_vision_data(matroot, split_file)
    T_matrix = load_data.prepare_wiki_data(npy_file)

    train(V_matrix_train, T_matrix, Y_matrix_train)

if __name__ == '__main__':
    test1()


