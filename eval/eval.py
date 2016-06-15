import tensorflow as tf
import numpy as np
import os
import sys
import math

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../inputs/')
sys.path.insert(0, '../model/')

import config
import bird_input
import architecture

eval_dir = config.eval_dir
checkpoint_dir = config.checkpoint_dir

def eval_once(saver, summary_writer, top_k_op, summary_op):
   print "Evaluating..."


def eval():
   with tf.Graph().as_default() as graph:

      images, labels = bird_input.inputs("test", 10, 1)

      logits = architecture.inference(images)

      # calculate predictions -> (predictions, targets, k)
      top_k_op = tf.nn.in_top_k(logits, labels, 1)

      sess = tf.Session()

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
      
      summary_writer = tf.train.SummaryWriter(eval_dir,  graph)     

      eval_once(saver, summary_writer, top_k_op, summary_op) 
   
      with tf.Session() as sess:
         # get the directory where models are stored
         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

         # restore a model
         saver.restore(sess, checkpoint_dir)

         # extract the global step
         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

         coord = tf.train.Coordinator()

         try:
            threads = []

            for q in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
               threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
               # change this later
               num_examples = 1139
               batch_size = 30
               num_iter = int(math.ceit(num_examples / batch_size))
               true_count = 0
               total_sample_count = num_iter * batch_size
               step = 0
               while step < num_iter:
                  predictions = sess.run([top_k_op])
                  true_count += np.sum(predictions)
                  step += 1

               precisions = true_count / total_sample_count
               print "precisions: " + str(precisions)
         except:
            exit()

         coord.request_stop()
         coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
   eval()

if __name__ == "__main__":
   tf.app.run()

