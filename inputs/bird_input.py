import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')
import config

data_dir = config.data_dir
dataset  = config.dataset

def read_records(filename_queue):

   class BirdRecord(object):
      pass
   result = BirdRecord()

   
   #result.height = 100
   #result.width  = 100
   #result.depth  = 3

   #image_bytes = result.height * result.width * result.depth

   #record_bytes = 

   reader = tf.TFRecordReader()

   
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
   image = tf.decode_raw(features['image'], tf.uint8)
   image.set_shape([100,100,3])

   # do some distortions here later

   label = tf.cast(featues['label'], tf.int32)

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

   image, label = read_and_decode(filename_queue)

   images, sparse_labels = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=2,
      capacity=1000+3*batch_size,
      min_after_dequeue=1000)

   return images, sparse_labels



