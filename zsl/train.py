__author__ = 'yuhongliang324'

import theano
import theano.tensor as T
from fc import FC
import timeit


def train(V_matrix, T_matrix, Y_matrix, obj='BCE', batch_size=200, max_epoch=100):

    mlp_t_layers, mlp_v_layers = [], []
    model = FC(mlp_t_layers, mlp_v_layers)
    symbols = model.define_functions(obj=obj)

    V_batch, T_batch, Y, updates = symbols['V_batch'], symbols['T_batch'], symbols['Y'], symbols['updates']
    is_train, cost, acc, pred = symbols['is_train'], symbols['cost'], symbols['acc'], symbols['pred']

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    print 'Compiling functions ... '
    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train, T_batch],
                            outputs=[pred, cost, acc], updates=updates,
                            givens={
                                V_batch: V_matrix[start_symbol: end_symbol + 1],
                                Y: Y_matrix[start_symbol: end_symbol + 1],
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
            start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, num_samples - 1)
            batch_index += 1

            pred, cost, acc = train_model(start, end, 1, T_matrix)
            cost_epoch += cost * (end - start + 1)
            acc_epoch += acc * (end - start + 1)

            print '\tAccuracy = %.4f\tTraining cost = %f' % (cost, acc)

            if end >= num_samples - 1:
                break

        cost_epoch /= num_samples
        acc_epoch /= num_samples

        print 'Epoch = %d\tAccuracy = %.4f\tTraining cost = %f' % (epoch_index + 1, acc_epoch, cost_epoch)

        end_time = timeit.default_timer()
        print 'Ran %.3f seconds for this epoch\n' % (end_time - start_time)
