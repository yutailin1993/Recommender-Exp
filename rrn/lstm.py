import tensorflow as tf


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
