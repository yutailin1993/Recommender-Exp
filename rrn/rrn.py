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
                    activation=tf.nn.relu)


class LSTM(object):
    def __init__(self, inputs, hparas):
        self.hparas = hparas
        self.inputs = inputs

        self._build_model()

    def _build_model(self):
        with tf.variable_scope('LSTM_'+self.hparas['NAME']):
            LSTMCell = tf.contrib.rnn.BasicLSTMCell(
                    self.hparas['LSTM_UNITS'],
                    activation=tf.nn.relu)
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
    def __init__(
            self, user_hparas, item_hparas, user_vectors, item_vectors,
            is_train, lr=0.01, epochs=100, loss_function='rmse'):
        self.user_hparas = user_hparas
        self.item_hparas = item_hparas
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        self.is_train = is_train
        self.lr = lr
        self.epochs = epochs
        self.loss_function = loss_function
        self.log = {'train_loss': []}
        self.turn = 1.  # 1 for 'user' turn, 0 for 'item' turn

        self._get_inputs()
        self._build_model()
        self._get_vars()
        self._get_loss()
        self._build_optimizer()
        self._get_session()
        self._get_saver()
        self._init_vars()

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

            self.logits = tf.nn.relu(self.logits)

    def _get_loss(self):
        if self.is_train:
            with tf.variable_scope('l2-regularizer'):
                user_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.user_vars])
                item_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.item_vars])

            with tf.variable_scope('loss'):
                if self.loss_function == 'rmse':
                    self.loss = tf.reduce_sum(tf.pow(self.ground_truth-self.logits, 2)) + \
                                    1 * self.turn * user_reg + \
                                    1 * (1-self.turn) * item_reg
                elif self.loss_function == 'log_loss':
                    self.loss = tf.reduce_sum(
                            -self.ground_truth*tf.log(self.logits)) + 0.01 * self.turn * user_reg + \
                                            0.01 * (1-self.turn) * item_reg
                else:
                    raise NotImplementedError
        else:
            with tf.variable_scope('loss'):
                self.loss = tf.sqrt(
                        tf.reduce_mean(tf.pow(self.ground_truth-self.logits, 2)))

    def _get_inputs(self):
        with tf.variable_scope('inputs'):
            self.user_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.user_hparas['BATCH_SIZE'],
                           None,
                           self.user_hparas['ITEM_NUM']),
                    name='user_in')
            self.item_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.item_hparas['BATCH_SIZE'],
                           None,
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
        if self.is_train:
            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.lr)
                # optimizer = tf.train.AdagradOptimizer(self.lr)

                self.user_optim = optimizer.minimize(
                        self.loss, var_list=self.user_vars)
                self.item_optim = optimizer.minimize(
                        self.loss, var_list=self.item_vars)

    def _get_vars(self):
        t_vars = tf.trainable_variables()

        self.user_vars = [var for var in t_vars if 'USER' in var.name]
        self.item_vars = [var for var in t_vars if 'ITEM' in var.name]

    def _get_session(self):
        self.sess = tf.Session()
        tf.summary.FileWriter('logs/', self.sess.graph)

    def _get_saver(self):
        self.saver = tf.train.Saver()

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self, df, user_map, item_map, initial_time):
        assert self.is_train is True
        prep = Preprocess(df, user_map, item_map, initial_time)

        for epoch in trange(self.epochs):
            loss = 0
            user_input, item_input, ground_truth, batch_user, batch_item = prep.gen_batch()
            u_static_vector = prep.get_latent_vector(batch_user, self.user_vectors, 'user')
            i_static_vector = prep.get_latent_vector(batch_item, self.item_vectors, 'item')

            # user turn
            self.turn = 1
            loss_, _ = self.sess.run(
                    [self.loss, self.user_optim],
                    feed_dict={
                        self.user_input: user_input,
                        self.item_input: item_input,
                        self.ground_truth: ground_truth,
                        self.user_stationary_factor: u_static_vector,
                        self.item_stationary_factor: i_static_vector,
                    })
            loss += loss_

            # item turn
            self.turn = 0
            loss_, _ = self.sess.run(
                    [self.loss, self.item_optim],
                    feed_dict={
                       self.user_input: user_input,
                       self.item_input: item_input,
                       self.ground_truth: ground_truth,
                       self.user_stationary_factor: u_static_vector,
                       self.item_stationary_factor: i_static_vector,
                    })
            loss += loss_

            self.log['train_loss'].append(loss/2)

    def model_saver(self, num):
        self.saver.save(self.sess, 'model/rrn_%d.ckpt' % (num + 1))

    def model_loader(self, num):
        self.saver.restore(self.sess, 'model/rrn_%d.ckpt' % (num))

    def test(self, df, user_map, item_map, initial_time):
        assert self.is_train is False
        prep = Preprocess(df, user_map, item_map, initial_time)
        userNum = len(np.unique(df['uid']))
        num_batch = userNum // self.user_hparas['BATCH_SIZE']

        test_loss = []
        for n in range(num_batch):
            user_input, item_input, ground_truth, \
                    batch_user, batch_item = prep.gen_batch(sector=n)

            u_static_vector = prep.get_latent_vector(batch_user, self.user_vectors, 'user')
            i_static_vector = prep.get_latent_vector(batch_item, self.item_vectors, 'item')

            loss_ = self.sess.run(
                    self.loss,
                    feed_dict={
                        self.user_input: user_input,
                        self.item_input: item_input,
                        self.ground_truth: ground_truth,
                        self.user_stationary_factor: u_static_vector,
                        self.item_stationary_factor: i_static_vector,
                    })
            test_loss.append(loss_)

        return test_loss

