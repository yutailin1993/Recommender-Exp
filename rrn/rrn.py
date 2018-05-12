import numpy as np
import tensorflow as tf
from preprocess import Preprocess
from tqdm import trange


class Transform(object):
    """Dimension transformation.

    Build model for different layers (Input embedded, rating 
    emission affine) transformation for both user and item.
    """

    def __init__(self, inputs, hparas, phase):
        self.inputs = inputs
        self.hparas = hparas
        self.phase = phase

        self._build_model()

    def _build_model(self):
        
        if self.phase == 'ENCODE':
            units = self.hparas['EMBED_UNITS']
            activate = tf.nn.sigmoid
        elif self.phase == 'AFFINE':
            units = self.hparas['LATENT_UNITS']
            activate = tf.nn.sigmoid

        with tf.variable_scope(self.phase+'_'+self.hparas['NAME']):
            self.output = tf.layers.dense(
                    self.inputs,
                    name='dense',
                    units=units,
                    activation=activate)


class LSTM(object):
    """LSTM Cell for learning temporal dynamics.

    In order to learn temporal dynamics, use LSTM Cell for both user
    and item. output is the predict output for all times and 
    output_last is for the new timestamp output.
    """

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
            self.output_last = output[:, -1, :]


class RRN(object):
    """Over all RRN model.

    Including dimension transform, LSTM and rating emission.
    """
    def __init__(
            self, user_hparas, item_hparas, user_vectors, item_vectors,
            is_train, lr=0.01, epochs=100, loss_function='rmse', weighted=None):
        self.user_hparas = user_hparas
        self.item_hparas = item_hparas
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        self.is_train = is_train
        self.lr = lr
        self.epochs = epochs
        self.loss_function = loss_function
        self.weighted = weighted
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
        """Build traning model.

        First embed user/item inputs into training dimension, then feed them in 
        LSTMCell. Further affine the matrics into emission dimension after LSTM
        and emit the prediction.
        """
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

            if self.loss_function == 'log_loss':
                logits = tf.add(
                        self.dynamic_state*0.5,
                        self.stationary_state*0.5,
                        name='logits')
                self.logits = tf.nn.sigmoid(logits, name='logits_activation')

            elif self.loss_function == 'rmse':
                logits = tf.add(
                        self.dynamic_state,
                        self.stationary_state,
                        name='logits')
                self.logits = tf.nn.relu(logits, name='logits_activation')
            else:
                raise NotImplementedError("Didn't implement the loss function yet.")

            self.logits_last = self.logits[-1, :, :]

    def _get_loss(self):
        """Get loss function.
        
        Get loss function with regularizer. There are two loss function can be 
        chosen, depending on the input data style(ether rating 1~5 or 0,1 for seen
        or unseen).
        """
        if self.is_train:
            with tf.variable_scope('l2-regularizer'):
                user_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.user_vars])
                item_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.item_vars])

            with tf.variable_scope('loss'):
                filter_ = tf.nn.relu(tf.sign(self.ground_truth[1:]))
                if self.weighted is None:
                    if self.loss_function == 'rmse':
                        self.loss = tf.reduce_sum(tf.multiply(tf.square(
                            tf.subtract(self.ground_truth[1:], self.logits[:-1])),
                            filter_)) + 1*self.turn*user_reg + 1*(1-self.turn)*item_reg

                    elif self.loss_function == 'log_loss':
                        self.loss = tf.reduce_mean(
                            -self.ground_truth[1:]*tf.log(self.logits[:-1]) -
                            (1-self.ground_truth[1:])*tf.log(1-self.logits[:-1])) + \
                            0.01 * self.turn * user_reg + \
                            0.01 * (1-self.turn) * item_reg
                    else:
                        raise NotImplementedError("Didn't implement the loss function yet.")
                else:
                    if self.loss_function == 'rmse':
                        weighted_loss = tf.multiply(tf.square(
                            tf.subtract(self.ground_truth[1:], self.logits[:-1])),
                            self.weight_list)
                        self.loss = tf.reduce_sum(tf.multiply(
                            weighted_loss,
                            filter_)) + 1*self.turn*user_reg + 1*(1-self.turn)*item_reg

                    elif self.loss_function == 'log_loss':
                        weighted_loss = tf.multiply(
                                -self.ground_truth[1:]*tf.log(self.logits[:-1]) -
                                (1-self.ground_truth[1:])*tf.log(1-self.logits[:-1]),
                                self.weight_list)
                        self.loss = tf.reduce_mean(weighted_loss) + \
                            0.01 * self.turn * user_reg + \
                            0.01 * (1-self.turn) * item_reg
                    else:
                        raise NotImplementedError("Didn't implement the loss function yet.")

        else:
            with tf.variable_scope('loss'):
                filter_ = tf.nn.relu(tf.sign(self.ground_truth[1:]))
                # use RMSE as prediction loss.
                if self.loss_function == 'rmse':
                    self.loss_indiv = tf.multiply(tf.square(
                        tf.subtract(self.ground_truth[1:], self.logits[:-1])),
                        filter_)
                    self.loss = tf.sqrt(tf.reduce_mean(tf.multiply(
                        tf.square(tf.subtract(self.ground_truth[1:], self.logits[:-1])),
                        filter_)))

                # use LOG_LOSS as prediction loss.
                elif self.loss_function == 'log_loss':
                    self.loss_indiv = -self.ground_truth[1:]*tf.log(self.logits[:-1]) - \
                        (1-self.ground_truth[1:])*tf.log(1-self.logits[:-1])
                    self.loss = tf.reduce_mean(
                        -self.ground_truth[1:]*tf.log(self.logits[:-1]) -
                        (1-self.ground_truth[1:])*tf.log(1-self.logits[:-1]))

    def _get_inputs(self):
        """Get input tensor.

        """
        with tf.variable_scope('inputs'):
            self.user_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.user_hparas['BATCH_SIZE'],
                           None,
                           self.user_hparas['VECTOR_LENGTH']),
                    name='user_in')
            self.item_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.item_hparas['BATCH_SIZE'],
                           None,
                           self.item_hparas['VECTOR_LENGTH']),
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

        if self.weighted is not None:
            with tf.variable_scope('weight_list'):
                self.weight_list = tf.placeholder(
                        dtype=tf.float32,
                        shape=(self.item_hparas['BATCH_SIZE']))

    def _build_optimizer(self):
        """Build optimizer.

        Use AdamOptimizer as paper said.
        """
        if self.is_train:
            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.lr)

                self.user_optim = optimizer.minimize(
                        self.loss, var_list=self.user_vars)
                self.item_optim = optimizer.minimize(
                        self.loss, var_list=self.item_vars)

    def _get_vars(self):
        """Get trainable variables.

        Get user and item's trainable variables to train them in different
        phase.
        """
        t_vars = tf.trainable_variables()

        self.user_vars = [var for var in t_vars if 'USER' in var.name]
        self.item_vars = [var for var in t_vars if 'ITEM' in var.name]

    def _get_session(self):
        """Get tensorflow session.

        """
        self.sess = tf.Session()
        tf.summary.FileWriter('logs/', self.sess.graph)

    def _get_saver(self):
        """Get model saver.

        """
        self.saver = tf.train.Saver()

    def _init_vars(self):
        """Initial all variables.

        """
        self.sess.run(tf.global_variables_initializer())

    def _run_session(self, *arg):
        """Run tf session.

        """
        raise NotImplementedError

    def train(self, df, user_map, item_map, initial_time, top_rank=None):
        """Train model.

        """
        assert self.is_train is True
        if self.loss_function == 'rmse':
            prep = Preprocess(
                    df, user_map, item_map, initial_time, 'rating', 
                    user_time_interval=self.user_hparas['TIME_INTERVAL'],
                    item_time_interval=self.item_hparas['TIME_INTERVAL'])
        elif self.loss_function == 'log_loss':
            prep = Preprocess(
                    df, user_map, item_map, initial_time, 'zero_one',
                    user_time_interval=self.user_hparas['TIME_INTERVAL'],
                    item_time_interval=self.item_hparas['TIME_INTERVAL'])

        if self.weighted is not None and top_rank is not None:
            weight_list = prep.get_list_weight(top_rank, self.weighted)
        elif self.weighted is not None and top_rank is None:
            raise ValueError("No ranking matrix.")

        for epoch in trange(self.epochs):
            loss = 0
            user_input, item_input, ground_truth, batch_user, batch_item = prep.gen_batch()

            u_static_vector = prep.get_latent_vector(batch_user, self.user_vectors, 'user')
            i_static_vector = prep.get_latent_vector(batch_item, self.item_vectors, 'item')
            ################################################################
            # without weighted
            ################################################################
            if self.weighted is None:
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
            
            ################################################################
            # With weighted
            ################################################################
            else:
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
                            self.weight_list: weight_list,
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
                           self.weight_list: weight_list,
                        })
                loss += loss_

            self.log['train_loss'].append(loss/2)

    def model_save(self, name):
        """Save model.

        """
        self.saver.save(self.sess, 'model/rrn_%s.ckpt' % (name))

    def model_load(self, name):
        """Load model.

        """
        self.saver.restore(self.sess, 'model/rrn_%s.ckpt' % (name))

    def test(self, df, user_map, item_map, initial_time,
             individually=False, top_rank=None):
        """Test model.

        """
        assert self.is_train is False

        if self.loss_function == 'rmse':
            prep = Preprocess(
                    df, user_map, item_map, initial_time, 'rating',
                    user_time_interval=self.user_hparas['TIME_INTERVAL'],
                    item_time_interval=self.item_hparas['TIME_INTERVAL'])
        elif self.loss_function == 'log_loss':
            prep = Preprocess(
                    df, user_map, item_map, initial_time, 'zero_one',
                    user_time_interval=self.user_hparas['TIME_INTERVAL'],
                    item_time_interval=self.item_hparas['TIME_INTERVAL'])

        if individually and top_rank is None:
            raise ValueError("No top_list!!")
        if individually and top_rank is not None:
            top_list = prep.get_top_list(top_rank)
        userNum = len(np.unique(df['uid']))
        num_batch = userNum // self.user_hparas['BATCH_SIZE']

        test_loss = []

        for n in range(num_batch):
            user_input, item_input, ground_truth, \
                    batch_user, batch_item = prep.gen_batch(sector=n)

            u_static_vector = prep.get_latent_vector(batch_user, self.user_vectors, 'user')
            i_static_vector = prep.get_latent_vector(batch_item, self.item_vectors, 'item')

            if individually:
                losses_ = self.sess.run(
                        self.loss_indiv,
                        feed_dict={
                            self.user_input: user_input,
                            self.item_input: item_input,
                            self.ground_truth: ground_truth,
                            self.user_stationary_factor: u_static_vector,
                            self.item_stationary_factor: i_static_vector,
                        })
                if self.loss_function == 'rmse':
                    loss_ = np.sqrt(np.mean(np.multiply(losses_, top_list)))
                elif self.loss_function == 'log_loss':
                    loss_ = np.mean(np.multiply(losses_, top_list))
            else:
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

    def predict(self):
        raise NotImplementedError
