"""

Cameron Fabbri
06/08/2016

Script that will create symbolic links to different files for train test and val.
The following folders are all contained in $DATA_DIR
A real directory structure is shown in tree.txt
tensorflow requires the folder structure to be:

   $TRAIN_DIR/dog/image0.jpeg
   $TRAIN_DIR/dog/image1.jpg
   $TRAIN_DIR/dog/image2.png
   ...
   $TRAIN_DIR/cat/weird-image.jpeg
   $TRAIN_DIR/cat/my-image.jpeg
   $TRAIN_DIR/cat/my-image.JPG
   ...
   $VALIDATION_DIR/dog/imageA.jpeg
   $VALIDATION_DIR/dog/imageB.jpg
   $VALIDATION_DIR/dog/imageC.png
   ...
   $VALIDATION_DIR/cat/weird-image.PNG
   $VALIDATION_DIR/cat/that-image.jpg
   $VALIDATION_DIR/cat/cat.JPG
   ...
   $TEST_DIR/dog/imageA.jpeg
   $TEST_DIR/dog/imageB.jpg
   $TEST_DIR/dog/imageC.png
   ...
   $TEST_DIR/cat/weird-image.PNG
   $VALIDATION_DIR/cat/that-image.jpg
   $VALIDATION_DIR/cat/cat.JPG
   Input is a root folder containing all image folders (data_dir)

   img_data_dir/
      - images
      - train
      - test
      - val

"""

import ntpath
import os
import sys
from random import shuffle
import math
import glob
import math

if __name__ == "__main__":

   # TODO make it so the user can delete the files in the test train val dirs if they run this script
   # twice, could be useful if they get more data or want to change the split size

   if len(sys.argv) < 2:
      print
      print "Usage: python create_test_train_val.py [dataset]"
      print "[dataset] is the name of your dataset (the directory name) in $DATA_DIR"
      print "Do not put a / at the end of the directory name"
      try:
         print "Current $DATA_DIR: " + str(os.environ['DATA_DIR'])
      except:
         print "$DATA_DIR not set. Set with `export DATA_DIR=/path/to/data_root` or put in ~/.bashrc"
      print
      exit()

   # split parameters
   train_perc = .80
   test_perc  = .1
   val_perc   = .1

   # name of the dataset, i.e the folder name in $DATA_DIR
   dataset = sys.argv[1]

   # the main directory containing all datasets
   data_dir  = os.environ['DATA_DIR']
  
   # the directory of the dataset we are using 
   dataset_dir = data_dir + "/" + dataset + "/images"

   train_dir = data_dir + "/" + dataset + "/train"
   test_dir  = data_dir + "/" + dataset + "/test" 
   val_dir   = data_dir + "/" + dataset + "/val" 

   label_list = list()

   # list of arrays in the form of one hot vectors
   hot_labels = list()

   for root, dirs, files in os.walk(dataset_dir):
      for label in dirs:
         label_list.append(label)
   
   # TODO create a mapping from label to one-hot vector
   # for now just find the position the label is in the label_list, and create that instance as the 1 in the vector

   # creating the test train and val directories
   try:
      os.mkdir(data_dir + "/" + dataset + "/test")
      os.mkdir(data_dir + "/" + dataset + "/train")
      os.mkdir(data_dir + "/" + dataset + "/val")
   except:
      pass
      #print "Directory already exists"

   # the label_list was created from the directory, so just go through that
   for label in label_list:
      # this will give the full path for the image in the folder
      image_list = glob.glob(dataset_dir+"/"+label+"/*.*")

      # this will just give the image names
      #image_list = os.listdir(dataset_dir+"/"+label)

      # shuffle the list so we get a random subset for train test val      
      shuffle(image_list)
      train_num = int(math.ceil(train_perc*len(image_list)))
      train_set = image_list[:train_num]

      test_num  = int(math.ceil(test_perc*len(image_list)))
      test_set  = image_list[train_num:train_num+test_num]
      val_set   = image_list[train_num+test_num:]

      # create a directory for the current label in test, val, and train
      try:
         os.mkdir(data_dir + "/" + dataset + "/train/" + label)
         os.mkdir(data_dir + "/" + dataset + "/test/" + label)
         os.mkdir(data_dir + "/" + dataset + "/val/" + label)
      except:
         continue   
     
      # now symlink the images in each list to their respective directory 
      for image in test_set:
         im_name = ntpath.basename(image)
         try:
            os.symlink(image, data_dir+"/"+dataset+"/test/"+label+"/"+im_name)
         except:
            continue
      
      for image in train_set:
         im_name = ntpath.basename(image)
         try:
            os.symlink(image, data_dir+"/"+dataset+"/train/"+label+"/"+im_name)
         except:
            continue

      for image in val_set:
         im_name = ntpath.basename(image)
         try:
            os.symlink(image, data_dir+"/"+dataset+"/val/"+label+"/"+im_name)
         except:
            continue




