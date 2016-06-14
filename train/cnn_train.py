import tensorflow as tf
import numpy as np
import os
import sys

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../inputs/')
sys.path.insert(0, '../model/')

import config
import bird_input
import architecture

def train():
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)

      images, labels = bird_input.inputs("train", 10, 1)
      print "Got images and labels"

      logits = architecture.inference(images)
      print "Got logits"

      loss = architecture.loss(logits, labels)
      print "Got loss"

      train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
      print "Made train_op"

      variables = tf.all_variables()
      print "Initialized variables"

      init = tf.initialize_all_variables()
      print "Initialized variables 2"

      sess = tf.Session()
      print "Made session"

      sess.run(init)
      print "Running session"

      tf.train.start_queue_runners(sess=sess)

      for i in xrange(1000):
         _, loss_value = sess.run([train_op, loss])
         print "Loss: " + str(loss_value)


def main(argv=None):
   train()

if __name__ == "__main__":
   tf.app.run()

