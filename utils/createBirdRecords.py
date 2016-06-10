"""
Cameron Fabbri

Creating three records, one train, one test, and one val, from the bird dataset

These records are generated from the $DATA_DIR. See the README for more info atm

"""

import tensorflow as tf
import numpy as np
import cv2
import os

def setup()
   # define the shape each image will be resized to
   SHAPE = (100, 100)

   # define output filenames for the records - good to put them in the $DATA_DIR
   try:
      data_dir = os.environ['DATA_DIR']
   except:
      print "$DATA_DIR not set"
      exit()

   # tf writer for train test and val
   train_writer = tf.python_io.TFRecordWriter(train_record)
   test_writer  = tf.python_io.TFRecordWriter(test_record)
   val_writer   = tf.python_io.TFRecordWriter(val_record)

# helper function
def _bytes_features(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



if __name__ == "__main__":
   setup()






