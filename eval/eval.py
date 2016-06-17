import tensorflow as tf
import numpy as np
import os
import sys
import math

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../inputs/')
sys.path.insert(0, '../model/')

import config
import input_
import architecture

eval_dir = config.eval_dir
checkpoint_dir = config.checkpoint_dir

# number of examples to run
num_examples = 1100
batch_size = config.batch_size

def eval():
   with tf.Graph().as_default() as graph:

      images, labels = input_.inputs("test", batch_size, 1)

      logits = architecture.inference(images, "train")

      # the in_top_k function requires the labels to be a vector of size batch_size
      # and for each element to be the label, so take the argmax for each
      labels = tf.argmax(labels, 1)

      # calculate predictions -> (predictions, targets, k)
      top_k_op = tf.nn.in_top_k(logits, labels, 1)

      sess = tf.Session()

      variables_to_restore = tf.all_variables()

      saver = tf.train.Saver(variables_to_restore)

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
      
      summary_writer = tf.train.SummaryWriter(eval_dir,  graph)     

      with tf.Session() as sess:
         # get the directory where models are stored
         
         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
         saver.restore(sess, ckpt.model_checkpoint_path)

         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
         coord = tf.train.Coordinator()

         try:
            tf.train.start_queue_runners(sess=sess)
            threads = []
            for q in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
               threads.extend(q.create_threads(sess, coord=coord, daemon=True, start=True))

               num_iter = int(math.ceil(num_examples / batch_size))
               true_count = 0
               total_sample_count = num_iter * batch_size
               step = 0
               while step < num_iter:
                  predictions = sess.run([top_k_op])
                  true_count += np.sum(predictions)
                  step += 1

            print "\n\n"
            print "true: " + str(true_count)
            print "total: " + str(total_sample_count)
            precisions = float(float(true_count) / float(total_sample_count))
            print str(precisions) + "% correct"
            print "\n\n"
         except Exception as e:
            print "\nFailed"
            print e
            print
            coord.request_stop(e)
            exit()

         coord.request_stop()
         coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
   eval()

if __name__ == "__main__":
   tf.app.run()

