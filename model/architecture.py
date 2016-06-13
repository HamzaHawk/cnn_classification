import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


# Constants describing the training process.

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")


def _variable_on_cpu(name, shape, initializer):
   with tf.device('/cpu:0'):
      var tf.get_variable(name, shape, initializer=initializer)
   return var


def inference(images):




