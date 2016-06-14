import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../utils/')
import config

FLAGS = tf.app.flags.FLAGS

num_epochs = 100

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")

data_dir = config.data_dir

def _variable_on_cpu(name, shape, initializer):
   with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
   return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var


def _activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inputs(data):
   type_input = "train"
   return bird_input.inputs(type_input, batch_size, num_epochs)

def maxpool2d(x, k=2):
   # MaxPool2D wrapper
   return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def _conv_layer(inputs, kernel_size, stride, num_features, idx):
   with tf.variable_scope('{0}_conv'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]

      weights = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_channels, num_features], stddev=0.1, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases', [num_features], tf.constant_initializer(0.1))

      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      #Leaky ReLU
      conv_rect = tf.maximum(FLAGS.alpha*conv_biased, conv_biased, name='{0}_conv'.format(idx))
      return conv_rect

def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('fc{0}'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs

    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')

    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.maximum(FLAGS.alpha*ip,ip,name=str(idx)+'_fc')


def inference(images):
           # input, kernel size, stride, num_features, num_epochs
   conv1 = _conv_layer(images, 5, 3, 32, 1)
   conv1 = maxpool2d(conv1, k=2)

   conv2 = _conv_layer(conv1, 2, 2, 32, 2)

   conv3 = _conv_layer(conv2, 5, 1, 64, 3)

   conv4 = _conv_layer(conv3, 2, 2, 32, 4)

   conv5 = _conv_layer(conv4, 2, 1, 32, 5)

   fc5 = _fc_layer(conv5, 512, 5, True, False)

   fc5_dropout = tf.nn.dropout(fc5, .5)

   # 200 is the number of classes
   y_1 = _fc_layer(fc5_dropout, 200, 6, False, False)

   y_1 = tf.nn.softmax(y_1)

   _activation_summary(y_1)

   return y_1


def loss (logits, labels):
  """ cross entropy loss by converting correcte_output to a one hot vector"""
  #correct_output_one_hot = architecture.one_hot(correct_output)
  error = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))
  return error


