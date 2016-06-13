import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')
import config

data_dir = config.data_dir
dataset  = config.dataset
   
def read_and_decode(filename_queue):

   reader = tf.TFRecordReader()
   _, serialized_example = reader.read(filename_queue)
   features = tf.parse_single_example(
      serialized_example,
      features={
         'image': tf.FixedLenFeature([], tf.string),
         'label': tf.FixedLenFeature([], tf.string)
      }
   )

   # convert from a scalar string tensor ot uint8 tensor
   # image = tf.decode_raw(features['image'], tf.)
   image = tf.decode_raw(features['image'], tf.float32)
   image = tf.reshape(image, [100,100,3])
   # do some distortions here later

   label = tf.decode_raw(features['label'], tf.float32)
   label = tf.reshape(label, [200])
   #exit()


   return image, label


def inputs(type_input, batch_size, num_epochs):
   if type_input == "train":
      filename = data_dir+"/"+dataset+"/records/train.tfrecord"
   elif type_input == "val":
      filename = data_dir+"/"+dataset+"/records/val.tfrecord"
   else:
      filename = data_dir+"/"+dataset+"/records/test.tfrecord"

   with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer( [filename], num_epochs=num_epochs)

   print "Reading data"
   image, label = read_and_decode(filename_queue)
   print "Read data"

   images, sparse_labels = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=2,
      capacity=1000+3*batch_size,
      min_after_dequeue=1000)

   return images, sparse_labels


def loss(logits, labels):
   """Add L2Loss to all the trainable variables.
   Add summary for "Loss" and "Loss/avg".
   Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
   Returns:
      Loss tensor of type float.
   """
   # Calculate the average cross entropy loss across the batch.
   labels = tf.cast(labels, tf.int64)
   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
   cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
   tf.add_to_collection('losses', cross_entropy_mean)

   # The total loss is defined as the cross entropy loss plus all of the weight
   # decay terms (L2 loss).
   return tf.add_n(tf.get_collection('losses'), name='total_loss')
