"""
Cameron Fabbri

Creating three records, one train, one test, and one val, from the bird dataset

These records are generated from the $DATA_DIR. See the README for more info atm

"""

import tensorflow as tf
import numpy as np
import config
import cv2
import os

# helper function
def _bytes_features(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def setup():
   # define the shape each image will be resized to
   SHAPE = (100, 100)

   data_dir = config.data_dir
   dataset  = config.dataset
   
   # create a records directory if there isn't one already
   try:
      os.mkdir(data_dir+"/"+dataset+"/records")
   except:
      pass

   # define output filenames for the records - good to put them in the $DATA_DIR
   train_record = data_dir+"/"+dataset+"/records/train.tfrecord"
   test_record  = data_dir+"/"+dataset+"/records/test.tfrecord"
   val_record   = data_dir+"/"+dataset+"/records/val.tfrecord"

   # tf writer for train test and val
   train_writer = tf.python_io.TFRecordWriter(train_record)
   test_writer  = tf.python_io.TFRecordWriter(test_record)
   val_writer   = tf.python_io.TFRecordWriter(val_record)

   #image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # loop through train test and val directories and write each image and label to the respective record
  
   # TODO need to convert labels to one-hot vectors, maybe have a function that does it or something

   t = ["/test/", "/val/", "/train/"]

   for a in t:
      for root, dirs, files in os.walk(data_dir+"/"+dataset+a):
         for d in dirs:
            label = d
            for r, l, images in os.walk(data_dir+"/"+dataset+a+label):
               for image in images:
                  img_src = data_dir+"/"+dataset+a+label+"/"+image
                  img = cv2.resize(img_src, SHAPE, interpolation = cv2.INTER_CUBIC)
                  example = tf.train.Example(features=tf.train.Features(feature={
                          'image': _bytes_feature(frame_raw),
                          'label': })) 


if __name__ == "__main__":
   setup()






