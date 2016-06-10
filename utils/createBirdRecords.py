"""
Cameron Fabbri

Creating three records, one train, one test, and one val, from the bird dataset
85:10:5 split of train:test:val

"""

import tensorflow as tf
import numpy as np

SHAPE = (100, 100)

def _bytes_features(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))








