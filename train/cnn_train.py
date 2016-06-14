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

checkpoint_dir = config.checkpoint_dir

"""

TODO: while training, use the eval record to periodically check how well the classifier
is doing. Write out the model at each x (whenever the eval is checked). At the end, 
use the model that did the best and test it on the test record

"""

def train():
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())

      images, labels = bird_input.inputs("train", 10, 1)

      logits = architecture.inference(images)

      loss = architecture.loss(logits, labels)

      train_op = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss)

      variables = tf.all_variables()

      init = tf.initialize_all_variables()

      sess = tf.Session()

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
      
      sess.run(init)
      print "Running session"

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      tf.train.start_queue_runners(sess=sess)

      for step in xrange(50000):
         _, loss_value = sess.run([train_op, loss])
         print "Loss: " + str(loss_value)

         # save for tensorboard
         if step%100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

         if step%1000 == 0:
            saver.save(sess, checkpoint_dir+"training", global_step=step)
            


def main(argv=None):
   train()

if __name__ == "__main__":
   tf.app.run()

