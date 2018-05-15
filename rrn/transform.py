import tensorflow as tf


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
