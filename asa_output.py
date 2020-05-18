import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


class ASAOutputLayer(object):
  """An output layer to predict Accessible Surface Area."""

  def __init__(self, name='asa'):
    self.name = name

  def compute_asa_output(self, activations):
    """Just compute the logits and outputs given activations."""
    asa_logits = tf.contrib.layers.linear(
        activations, 1,
        weights_initializer=tf.random_uniform_initializer(-0.01, 0.01),
        scope='ASALogits')
    self.asa_output = tf.nn.relu(asa_logits, name='ASA_output_relu')

    return asa_logits