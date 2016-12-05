__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


class FC(object):

    def __init__(self, mlp_t_layers, mlp_v_layers, word_dim=0, hid_dim=0, lamb=0.0001, drop=0.5, update='rmsprop',
                 lr=None, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0., momentum=0.9, rho=0.9):

        self.mlp_t_layers, self.mlp_v_layers = mlp_t_layers, mlp_v_layers
        self.word_dim, self.hid_dim = word_dim, hid_dim
        self.lamb = lamb
        self.drop = drop
        self.update = update
        self.lr, self.momentum, self.rho = lr, momentum, rho
        self.beta1, self.beta2, self.epsilon, self.decay = beta1, beta2, epsilon, decay
        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.name = 'fc'

        self.W_t_mlp, self.W_v_mlp = [], []
        self.b_t_mlp, self.b_v_mlp = [], []
        self.theta = []

        self.initialize_mlp_layers(self.mlp_t_layers, self.W_t_mlp, self.b_t_mlp)
        self.initialize_mlp_layers(self.mlp_v_layers, self.W_v_mlp, self.b_v_mlp)

        # LSTM parameters
        self.W_i, self.b_i = self.init_para(self.word_dim, self.hid_dim)
        self.U_i, _ = self.init_para(self.hid_dim, self.hid_dim)
        self.W_f, self.b_f = self.init_para(self.word_dim, self.hid_dim)
        self.U_f, _ = self.init_para(self.hid_dim, self.hid_dim)
        self.W_o, self.b_o = self.init_para(self.word_dim, self.hid_dim)
        self.U_o, _ = self.init_para(self.hid_dim, self.hid_dim)
        self.W_c, self.b_c = self.init_para(self.word_dim, self.hid_dim)
        self.U_c, _ = self.init_para(self.hid_dim, self.hid_dim)

        self.add_param_shapes()

        if self.update == 'adagrad':
            if lr:
                self.lr = lr
            else:
                self.lr = 0.01
            self.grad_histories = [
                theano.shared(
                    value=numpy.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                    name="grad_hist:" + param.name
                )
                for param_shape, param in zip(self.param_shapes, self.theta)
                ]
        elif self.update == 'sgdm':
            if lr:
                self.lr = lr
            else:
                self.lr = 0.01
            self.velocity = [
                theano.shared(
                    value=numpy.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                    name="momentum:" + param.name
                )
                for param_shape, param in zip(self.param_shapes, self.theta)
                ]
            self.momentum = momentum
        elif self.update == 'rmsprop' or self.update == 'RMSprop':
            self.rho = rho
            if lr:
                self.lr = lr
            else:
                self.lr = 0.001
            self.weights = [
                theano.shared(
                    value=numpy.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                )
                for param_shape, param in zip(self.param_shapes, self.theta)
                ]
        elif self.update == 'adam' or self.update == 'Adam':
            pass
        else:  # sgd
            if lr:
                self.lr = lr
            else:
                self.lr = 0.01

    def initialize_mlp_layers(self, mlp_layers, Ws, bs):
        num_layers = len(mlp_layers)
        for i in xrange(num_layers - 1):
            d1, d2 = mlp_layers[i], mlp_layers[i + 1]
            W_values = numpy.asarray(self.rng.uniform(
                       low=-numpy.sqrt(6. / float(d1 + d2)), high=numpy.sqrt(6. / float(d1 + d2)), size=(d1, d2)),
                       dtype=theano.config.floatX)
            W = theano.shared(value=W_values, borrow=True)
            b_values = numpy.zeros((d2,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
            Ws.append(W)
            bs.append(b)

            self.theta += [W, b]

    def init_para(self, d1, d2):
        if d1 + d2 == 0:
            low, high = 0, 0
        else:
            high = numpy.sqrt(6. / float(d1 + d2))
            low = - high
        W_values = numpy.asarray(self.rng.uniform(
            low=low, high=high, size=(d1, d2)),
            dtype=theano.config.floatX)
        W = theano.shared(value=W_values, borrow=True)
        b_values = numpy.zeros((d2,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        self.theta += [W, b]

        return W, b

    def add_param_shapes(self):
        self.param_shapes = []
        for param in self.theta:
            self.param_shapes.append(param.get_value().shape)

    def l2(self):
        l2 = self.lamb * T.sum([T.sum(p ** 2) for p in self.theta])
        return l2

    def dropout(self, layer, is_train):
        mask = self.theano_rng.binomial(p=self.drop, size=layer.shape, dtype=theano.config.floatX)
        return T.cast(T.switch(T.neq(is_train, 0), layer * mask, layer * self.drop), dtype=theano.config.floatX)

    # scan function parameter order: sequences, prior results, non_sequences
    # sequences: X_t (num_train_class, word_dim)
    # prior results: C_tm1, H_tm1 (batch_size, hid_dim)
    def forward(self, X_t, C_tm1, H_tm1):
        i_t = T.nnet.sigmoid(T.dot(X_t, self.W_i) + T.dot(H_tm1, self.U_i) + self.b_i)  # (num_train_class, hid_dim)
        f_t = T.nnet.sigmoid(T.dot(X_t, self.W_f) + T.dot(H_tm1, self.U_f) + self.b_f)  # (num_train_class, hid_dim)
        o_t = T.nnet.sigmoid(T.dot(X_t, self.W_o) + T.dot(H_tm1, self.U_o) + self.b_o)  # (num_train_class, hid_dim)
        C_t = T.tanh(T.dot(X_t, self.W_c) + T.dot(H_tm1, self.U_c) + self.b_c)  # (num_train_class, hid_dim)
        C_t = i_t * C_t + f_t * C_tm1  # (num_train_class, hid_dim)
        H_t = o_t * T.tanh(C_t)  # (num_train_class, hid_dim)
        return C_t, H_t

    # For BCE, Y is 0-1 encoded; for hinge loss, Y is {-1, 1} encoded
    # Use minibatch classes instead of all classes
    def define_functions(self, obj='BCE'):

        V_batch = T.matrix()  # (batch_size, d)
        T_batch = T.matrix()  # (num_train_class, p)
        Y_batch = T.matrix()  # (batch_size, num_train_class)

        is_train = T.iscalar()

        V_rep = V_batch
        for i in xrange(len(self.W_v_mlp)):
            W, b = self.W_v_mlp[i], self.b_v_mlp[i]
            V_rep = T.tanh(T.dot(V_rep, W) + b)
            # V_rep = self.dropout(V_rep, is_train)

        S_batch = T.tensor3()  # (n_step, num_train_class, word_dim)
        num_train_class = T.shape(S_batch)[1]
        [_, H], _ = theano.scan(self.forward, sequences=S_batch,
                                outputs_info=[T.zeros((num_train_class, self.hid_dim), dtype=theano.config.floatX),
                                              T.zeros((num_train_class, self.hid_dim), dtype=theano.config.floatX)])
        rep = H[-1]  # (num_train_class, hid_dim)
        w = T.concatenate((T_batch, rep), axis=1)

        for i in xrange(len(self.W_t_mlp)):
            W, b = self.W_t_mlp[i], self.b_t_mlp[i]
            w = T.tanh(T.dot(w, W) + b)
            # w = self.dropout(w, is_train)

        # V_rep: (batch_size, k)
        # w: (num_train_class, k)

        sim = T.dot(V_rep, w.T)  # (batch_size, num_train_class)

        if obj == 'BCE' or obj == 'bce':
            sim = T.nnet.sigmoid(sim)
            loss = - T.mean(T.sum(Y_batch * T.log(sim) + (1. - Y_batch) * T.log(1. - sim), axis=1))
        elif obj == 'Hinge' or obj == 'hinge':
            loss = T.mean(T.sum(T.maximum(0, 1. - Y_batch * sim), axis=1))
        else:
            V_norm = T.sum(V_rep * V_rep, axis=1)  # (batch_size,)
            w_norm = T.sum(w * w, axis=1)  # (num_train_class,)
            sim -= w_norm / 2.
            sim = (sim.T - V_norm / 2.).T
            loss = T.mean(T.sum(T.maximum(0, 1. - Y_batch * sim), axis=1))

        pred = T.argmax(sim, axis=1)
        labels = T.argmax(Y_batch, axis=1)
        acc = T.mean(T.eq(pred, labels))

        cost = loss + self.l2()
        gradients = [T.grad(cost, param) for param in self.theta]

        # adagrad
        if self.update == 'adagrad':
            new_grad_histories = [
                T.cast(g_hist + g ** 2, dtype=theano.config.floatX)
                for g_hist, g in zip(self.grad_histories, gradients)
                ]
            grad_hist_update = zip(self.grad_histories, new_grad_histories)

            param_updates = [(param, T.cast(param - self.lr / (T.sqrt(g_hist) + self.epsilon) * param_grad, dtype=theano.config.floatX))
                             for param, param_grad, g_hist in zip(self.theta, gradients, new_grad_histories)]
            updates = grad_hist_update + param_updates
        # SGD with momentum
        elif self.update == 'sgdm':
            velocity_t = [self.momentum * v + self.lr * g for v, g in zip(self.velocity, gradients)]
            velocity_updates = [(v, T.cast(v_t, theano.config.floatX)) for v, v_t in zip(self.velocity, velocity_t)]
            param_updates = [(param, T.cast(param - v_t, theano.config.floatX)) for param, v_t in zip(self.theta, velocity_t)]
            updates = velocity_updates + param_updates
        elif self.update == 'rmsprop' or self.update == 'RMSprop':
            updates = []
            for p, g, a in zip(self.theta, gradients, self.weights):
                # update accumulator
                new_a = self.rho * a + (1. - self.rho) * T.square(g)
                updates.append((a, new_a))
                new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
                updates.append((p, new_p))
        # basic SGD
        else:
            updates = OrderedDict((p, T.cast(p - self.lr * g, dtype=theano.config.floatX)) for p, g in zip(self.theta, gradients))

        ret = {'V_batch': V_batch, 'T_batch': T_batch, 'Y_batch': Y_batch, 'S_batch': S_batch,
                'updates': updates, 'is_train': is_train, 'cost': cost, 'loss': loss,
                'acc': acc, 'pred': pred, 'sim': sim}
        return ret
