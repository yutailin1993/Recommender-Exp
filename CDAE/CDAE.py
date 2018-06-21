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
            batch_size=64, loss_function='rmse', b1=0.5, optimizer='adagrad'):
        '''
        -- Args --
            user_num: number of users
            item_num: number of items
            mode: "user" or "item"
            denoising: "False" if want inputs itself be targets
                       otherwise, denoise AutoEncoder
            lr: learning rate
            hidden_units: number of middle layers units
            epochs: number of learning epochs
            batch_size: mini batch size
            b1: beta1, for adadelta optimizer
            optimizer: specify which optimizer to use
            loss_function: specify which loss function to use
            is_cluster: specify if is in cluster mode
        '''

        self.user_num = user_num
        self.item_num = item_num
        self.mode = mode
        self.dropout_rate = dropout_rate
        self.b1 = b1
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.with_weights = with_weights
        self.hidden_units = hidden_units
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.log = {'train_loss': [], 'ap@5': [], 'ap@10': [], 'recall@5': [], 'recall@10': []}

        if self.mode != 'user' and self.mode != 'item':
            print (self.mode)
            raise ValueError

        self._get_inputs()
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self._build_session()
        self._get_saver()
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

            self.ident = tf.placeholder(tf.int32, shape=[None])

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
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                    name='encode')

            self.code = tf.nn.sigmoid(
                    tf.add(self.encode, self.specVector),
                    name='add_specVector')

            self.decode = tf.layers.dense(
                    inputs=self.code,
                    units=dec_units,
                    activation=tf.nn.sigmoid,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                    name='decode')

    def _build_loss(self):
        with tf.variable_scope('loss'):
            if self.loss_function == 'rmse':
                self.loss = tf.sqrt(
                        tf.reduce_mean(tf.pow(self.target-self.decode, 2)))

            elif self.loss_function == 'cross_entropy':
                self.loss = tf.reduce_mean(
                        tf.reduce_sum(-self.target*tf.log(self.decode) - \
                                (1-self.target)*tf.log(1-self.decode),
                                reduction_indices=1))

            elif self.loss_function == 'log_loss':
                self.loss = tf.losses.log_loss(self.target, self.logits)

            else:
                raise NotImplementedError

    def _get_inputs(self):
        if self.mode == 'user':
            self.input = tf.placeholder(tf.float32,
                    shape=(None, self.item_num), name='input')
            self.target = tf.placeholder(tf.float32,
                    shape=(None, self.item_num), name='target')
        elif self.mode == 'item':
            self.input = tf.placeholder(tf.float32,
                    shape=(None, self.user_num), name='input')
            self.target = tf.placeholder(tf.float32,
                    shape=(None, self.user_num), name='target')

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

    def _get_saver(self):
        self.saver = tf.train.Saver()

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self, rating, train_idents, train_indices=None, test_indices=None):
        num_batch = len(train_idents) // self.batch_size
        train_idents_idx = [k for k in range(len(train_idents))]

        for epoch in trange(self.epochs):
            total_loss = 0
            ap_at_5 = []
            ap_at_10 = []
            recall_at_5 = []
            recall_at_10 = []

            for n in range(num_batch+1):
                if n < num_batch:
                    valid_num = self.batch_size
                else:
                    valid_num = len(train_idents) - n * self.batch_size

                if valid_num == 0:
                    break

                start = n * self.batch_size
                # inputs = np.take(rating, train_idents_idx[start: start+valid_num], axis=0)

                # input_, train_indices, test_indices = gen_train_test(inputs)
                input_ = np.take(rating, train_idents_idx[start: start+valid_num], axis=0)
                target_ = input_
                idents_ = train_idents[start: start+valid_num]

                recon, loss, _ = self.sess.run(
                        [self.decode, self.loss, self.optim],
                        feed_dict={
                            self.input: input_,
                            self.target: target_,
                            self.ident: idents_,
                        })

                total_loss += loss

                if epoch % (self.epochs*0.01) == 0 and epoch > 0:
                    top10 = get_topN(recon, train_indices[start: start+valid_num], N=10)
                    top5 = get_topN(recon, train_indices[start: start+valid_num], N=5)
                    iAP_5 = avg_precision(top5, test_indices[start: start+valid_num])
                    iAP_10 = avg_precision(top10, test_indices[start: start+valid_num])
                    iRecall_5 = recall_at_N(top5, test_indices[start: start+valid_num], N=5)
                    iRecall_10 = recall_at_N(top10, test_indices[start: start+valid_num], N=10)

                    if iAP_5 is not None:
                        ap_at_5.append(iAP_5)
                    if iAP_10 is not None:
                        ap_at_10.append(iAP_10)
                    if iRecall_5 is not None:
                        recall_at_5.append(iRecall_5)
                    if iRecall_10 is not None:
                        recall_at_10.append(iRecall_10)

            self.log['train_loss'].append(total_loss/len(train_idents))

            if epoch % (self.epochs*0.01) == 0 and epoch > 0:
                self.log['ap@5'].append(sum(ap_at_5)/len(ap_at_5))
                self.log['ap@10'].append(sum(ap_at_10)/len(ap_at_10))
                self.log['recall@5'].append(sum(recall_at_5)/len(recall_at_5))
                self.log['recall@10'].append(sum(recall_at_10)/len(recall_at_10))

    def train_all(self, rating, train_idents):
        """Train with all rating without validation

        """

        num_batch = len(train_idents) // self.batch_size
        train_idents_idx = [k for k in range(len(train_idents))]

        for epoch in trange(self.epochs):
            total_loss = 0
            
            for n in range(num_batch+1):
                if n < num_batch:
                    valid_num = self.batch_size
                else:
                    valid_num = len(train_idents) - n * self.batch_size

                if valid_num == 0:
                    break

                start = n * self.batch_size
                input_ = np.take(rating, train_idents_idx[start: start+valid_num], axis=0)
                target_ = input_
                idents_ = train_idents[start: start+valid_num]

                recon, loss, _ = self.sess.run(
                        [self.decode, self.loss, self.optim],
                        feed_dict={
                            self.input: input_,
                            self.target: target_,
                            self.ident: idents_,
                        })

                total_loss += loss

            self.log['train_loss'].append(total_loss/len(train_idents))

    def model_save(self, num):
        self.saver.save(self.sess, 'model/cdae_%d.ckpt' % (num))

    def model_load(self, num):
        self.saver.restore(self.sess, 'model/cdae_%d.ckpt' % (num))

    def predict(self, rating, test_idents):
        # recon_list = []

        num_batch = len(test_idents) // self.batch_size
        test_idents_idx = [k for k in range(len(test_idents))]

        recon_list = np.zeros(shape=(rating.shape[0], rating.shape[1]), dtype=np.float32)

        for n in range(num_batch+1):
            if n < num_batch:
                valid_num = self.batch_size
            else:
                valid_num = len(test_idents) - n * self.batch_size

            if valid_num == 0:
                break

            start = n * self.batch_size
            input_ = np.take(rating, test_idents_idx[start: start+valid_num], axis=0)
            idents_ = test_idents[start: start+valid_num]

            recon = self.sess.run(
                    self.decode,
                    feed_dict={
                        self.input: input_,
                        self.ident: idents_
                    })

            # recon_list.append(recon)
            recon_list[start: start+valid_num, :] = recon

        # recon_list = np.squeeze(np.asarray(recon_list), axis=0)

        return recon_list
