## Scripts for handling data in tensorflow

#### Important
update: config.py contains variables specific to this app. This is so these scripts can be easily
copied over to another dataset or project, also less messy than using an environment variable

### `create_test_train_val.py`
This script takes in a root data folder containing images labeled by the subdirectory they are in,
and seperates them into test train and val sets using symlinks. This is so tensorflow can create
a tfrecord for each set.

$DATA_DIR is defined as an environment variable in ~/.bashrc. This script will take an argument to
use the data from that dataset. For example, if there is a dataset mnist in the data_dir like so:

data_dir:
   - mnist
      - README
      - attributes
      - images
         - label1
         - label2
         - ...

the script will be called as `python create_test_train_val.py mnist` and automatically look for a folder
called `images` in the directory containing folders with images in them. Each folder is the label for those
images.

The symlink images the script will create are located in the top dataset directory under train, test, and val.
For mnist this would look like:

data_dir:
   - mnist
      - test
         -label1
         -label2
         - ...
      - train
         -label3
         -label4
         - ...
      - val
         -label5
         -label6
         - ...
      - README
      - attributes
      - images
         - label1
         - label2
         - ...


label must be a one-hot vector

label will be from _bytes_feature

file structure:
   main file: structure of the model architecture
   input file: reads the tf record



