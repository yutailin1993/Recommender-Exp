import numpy as np
import tensorflow as tf
from preprocess import Preprocess
from tqdm import trange


class Transform(object):
    def __init__(self, inputs, hparas, phase):
        self.inputs = inputs
        self.hparas = hparas
        self.phase = phase

        self._build_model()

    def _build_model(self):
        
        if self.phase == 'ENCODE':
            units = self.hparas['EMBED_UNITS']
        elif self.phase == 'AFFINE':
            units = self.hparas['LATENT_UNITS']

        with tf.variable_scope(self.phase+'_'+self.hparas['NAME']):

            self.output = tf.layers.dense(
                    self.inputs,
                    name='dense',
                    units=units,
                    activation=tf.nn.sigmoid)


class LSTM(object):
    def __init__(self, inputs, hparas):
        self.hparas = hparas
        self.inputs = inputs

        self._build_model()

    def _build_model(self):
        with tf.variable_scope('LSTM_'+self.hparas['NAME']):
            LSTMCell = tf.contrib.rnn.BasicLSTMCell(
                    self.hparas['LSTM_UNITS'])
            initial_state = LSTMCell.zero_state(
                    self.hparas['BATCH_SIZE'], dtype=tf.float32)
            output, last_state = tf.nn.dynamic_rnn(
                    cell=LSTMCell,
                    inputs=self.inputs,
                    initial_state=initial_state,
                    dtype=tf.float32,
                    )

            self.output = output
            self.last_state = last_state


class RRN(object):
    def __init__(self, user_hparas, item_hparas, lr):
        self.user_hparas = user_hparas
        self.item_hparas = item_hparas
        self.lr = lr
        
        self._get_inputs()
        self._build_model()
        self._get_vars()
        self._build_optimizer()
        self._get_session()

    def _build_model(self):

        phase = 'ENCODE'
        with tf.variable_scope(phase):
            self.encode_user = Transform(self.user_input, self.user_hparas, phase)
            self.encode_item = Transform(self.item_input, self.item_hparas, phase)

        phase = 'LSTM'
        with tf.variable_scope(phase):
            self.lstm_user = LSTM(self.encode_user.output, self.user_hparas)
            self.lstm_item = LSTM(self.encode_item.output, self.item_hparas)

        phase = 'AFFINE'
        with tf.variable_scope(phase):
            self.trans_user = Transform(self.lstm_user.output, self.user_hparas, phase)
            self.trans_item = Transform(self.lstm_item.output, self.item_hparas, phase)

        phase = 'EMISSION'
        with tf.variable_scope(phase):
            self.dynamic_state = tf.einsum(
                    'ijl,kjl->jik',
                    self.trans_user.output,
                    self.trans_item.output,
                    name='dynamic_state')
            self.stationary_state = tf.matmul(
                    self.user_stationary_factor,
                    self.item_stationary_factor,
                    transpose_b=True,
                    name='stationary_state')
            self.logits = tf.add(
                    self.dynamic_state,
                    self.stationary_state,
                    name='logits')

        with tf.variable_scope('loss'):
            loss_ = tf.subtract(self.ground_truth, self.logits)
            self.loss = tf.reduce_sum(tf.square(loss_))

    def _get_inputs(self):
        with tf.variable_scope('inputs'):
            self.user_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.user_hparas['BATCH_SIZE'],
                        self.user_hparas['TIME_ELAPSE'],
                        self.user_hparas['ITEM_NUM']),
                    name='user_in')
            self.item_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.item_hparas['BATCH_SIZE'],
                        self.item_hparas['TIME_ELAPSE'],
                        self.item_hparas['USER_NUM']),
                    name='item_in')

        with tf.variable_scope('stationary_factor'):
            self.user_stationary_factor = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.user_hparas['BATCH_SIZE'],
                        self.user_hparas['STATIONARY_LENGTH']),
                    name='user_stationary')
            self.item_stationary_factor = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.item_hparas['BATCH_SIZE'],
                        self.item_hparas['STATIONARY_LENGTH']),
                    name='item_stationary')

        with tf.variable_scope('ground_truth'):
            self.ground_truth = tf.placeholder(
                    dtype=tf.float32,
                    shape=(None, self.user_hparas['BATCH_SIZE'],
                        self.item_hparas['BATCH_SIZE']))

    def _build_optimizer(self):
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            
            self.user_optim = optimizer.minimize(
                    self.loss, var_list=self.user_vars)
            self.item_optim = optimizer.minimize(
                    self.loss, var_list=self.item_vars)

    def _get_vars(self):
        t_vars = tf.trainable_variables()

        self.user_vars = [var for var in t_vars if 'USER' in var.name]
        self.item_vars = [var for var in t_vars if 'ITEM' in var.name]

    def train(self, df):
        prep = Preprocess(df)

        losses = []
        for epoch in trange(1000):
            user_inputs, item_inputs, ground_truth, _, _ = prep.gen_batch()

            # user turn
            loss_, _ = self.sess.run(
                    [self.loss, self.user_optim],
                    feed_dict={self.})


    def _get_session(self):
        self.sess = tf.Session()

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def model_saver(self):
        raise NotImplementedError

    def model_loader(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError

