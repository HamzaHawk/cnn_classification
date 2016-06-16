import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')
import config

data_dir = config.data_dir
dataset  = config.dataset
num_classes = config.num_classes

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

   image = tf.decode_raw(features['image'], tf.uint8)
   image = tf.to_float(image, name='float32')
   
   image = tf.reshape(image, [100,100,3])
   # do some distortions here later

   label = tf.decode_raw(features['label'], tf.float64)
   label = tf.to_float(label, name='float32')
   #label = tf.to_float(label, name='int32')
   label = tf.reshape(label, [num_classes])

   return image, label


def inputs(type_input, batch_size, num_epochs):
   if type_input == "train":
      filename = data_dir+"/"+dataset+"/records/train.tfrecord"
   elif type_input == "val":
      filename = data_dir+"/"+dataset+"/records/val.tfrecord"
   else:
      filename = data_dir+"/"+dataset+"/records/test.tfrecord"

   filename_queue = tf.train.string_input_producer([filename])

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


