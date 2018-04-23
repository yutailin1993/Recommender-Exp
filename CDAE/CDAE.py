import tensorflow as tf
from tqdm import trange
import numpy as np
from utils import *


class AutoEncoder(object):
    '''
    -- CDAE --
    Discription:
        Try to learn latent representation from rating matrix
    '''

    def __init__(
            self, user_num, item_num, mode, with_weights=False,
            dropout_rate=0.2, lr=0.01, hidden_units=20, epochs=100,
            loss_function='rmse', b1=0.5, optimizer='adagrad'):
        '''
        -- Args --
            user_num: number of users
            item_num: number of items
            mode: "user" or "item"
            denoising: "False" if want inputs itself be targets
                       otherwise, denoise AutoEncoder
            lr: learning rate
            hidden_units: number of middle layers units
            epochs: number of learning epoch
            b1: beta1, for adadelta optimizer
            optimizer: specify which optimizer to use
            loss_function: specify which loss function to use
        '''

        self.user_num = user_num
        self.item_num = item_num
        self.mode = mode
        self.dropout_rate = dropout_rate
        self.b1 = b1
        self.lr = lr
        self.epochs = epochs
        self.with_weights = with_weights
        self.hidden_units = hidden_units
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.log = {'train_loss': [], 'ap@5': []}

        if self.mode != ('user' or 'item'):
            print (self.mode)
            raise ValueError

        self._get_inputs()
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self._build_session()
        self._init_vars()
        
    def _build_model(self):
        hidden_units = self.hidden_units
        user_num = self.user_num
        item_num = self.item_num

        if self.mode == 'user':
            vec_shape = (user_num, hidden_units)
            dec_units = item_num
        elif self.mode == 'item':
            vec_shape = (item_num, hidden_units)
            dec_units = user_num

        # ======================================================================
        # build user/item specify vector
        # ======================================================================
        with tf.variable_scope('specify_vector'):
            
            self.vector_matrix = tf.get_variable(
                    self.mode+'_vector',
                    shape=vec_shape,
                    initializer=tf.random_normal_initializer(stddev=0.5),
                    dtype=tf.float32)

            self.ident = tf.placeholder(tf.int32, shape=[])

            self.specVector = tf.nn.embedding_lookup(self.vector_matrix, self.ident)

        # ======================================================================
        # denoising
        # ======================================================================
        with tf.variable_scope('denoising'):
            self.noise_input = tf.nn.dropout(self.input, 1-self.dropout_rate)

        # ======================================================================
        # Autoencoder
        # ======================================================================
        with tf.variable_scope('AutoEncoder'):
            self.encode = tf.layers.dense(
                    inputs=self.noise_input,
                    units=hidden_units,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0003),
                    name='encode')

            self.code = tf.nn.sigmoid(
                    tf.add(self.encode, self.specVector),
                    name='add_specVector')

            self.decode = tf.layers.dense(
                    inputs=self.code,
                    units=dec_units,
                    activation=tf.nn.sigmoid,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0003),
                    name='decode')

    def _build_loss(self):
        with tf.variable_scope('loss'):
            if self.loss_function == 'rmse':
                self.loss = tf.sqrt(
                        tf.reduce_mean(tf.pow(self.target-self.decode, 2)))

            elif self.loss_function == 'log_loss':
                self.loss = tf.reduce_mean(
                        tf.reduce_sum(-self.target*tf.log(self.decode) - \
                                (1-self.target)*tf.log(1-self.decode),
                                reduction_indices=1))
            else:
                raise NotImplementedError

    def _get_inputs(self):
        if self.mode == 'user':
            self.input = tf.placeholder(tf.float32,
                    shape=(1, self.item_num), name='input')
            self.target = tf.placeholder(tf.float32,
                    shape=(1, self.item_num), name='target')
        elif self.mode == 'item':
            self.input = tf.placeholder(tf.float32,
                    shape=(1, self.user_num), name='input')
            self.target = tf.placeholder(tf.float32,
                    shape=(1, self.user_num), name='target')

    def _build_optimizer(self):
        '''
        Initialize optimizer
        '''
        if self.optimizer == 'adadelta':
            self.optim = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer == 'gradient':
            self.optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer == 'adagrad':
            self.optim = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        else:
            raise NotImplementedError

    def _build_session(self):
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('logs/', self.sess.graph)

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self, rating, train_indices, test_indices):
        if self.mode == 'user':
            train_num = self.user_num
        elif self.mode == 'item':
            train_num = self.item_num
        
        for epoch in trange(self.epochs):
            total_loss = 0
            ap_at_5 = []

            for n in range(train_num):
                input_ = [rating[n]]
                target_ = [rating[n]]

                loss, _ = self.sess.run(
                        [self.loss, self.optim],
                        feed_dict={
                            self.input: input_,
                            self.target: target_,
                            self.ident: n,
                        })

                recon = self.decode.eval(
                        session=self.sess,
                        feed_dict={
                            self.input: input_,
                            self.ident: n
                        })

                total_loss += loss
                top5 = get_topN(recon, train_indices[n])
                ap_at_5.append(avg_precision(top5, test_indices[n]))

            self.log['train_loss'].append(total_loss/train_num)
            self.log['ap@5'].append(sum(ap_at_5)/len(ap_at_5))

