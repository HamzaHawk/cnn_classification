## Image Classification using a Convolutional Neural Network

This attemps to classify different types of birds coming from the [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset.
The dataset includes 200 classes of different birds. The implementation uses [Tensorflow](http://tensorflow.org/) compiled for use on a GPU. Rudementary training
is complete, although validation is not.

### Data Management
The scripts in the `utils` folder help organize the data. The data must be stored in a directory (known as `data_dir`)
that is specified in `utils/config.py`. The variables defined in `config.py` are also used in other files throughout
the project. In short, in `data_dir` there should be folder seperated datasets, each with a folder called `images` that
contains folder seperated sets of images, the folder mapping to the label for the images in it. Below is an example for
how the `mnist` dataset would be structured.


data_dir:
   - mnist:
      - README.txt
      - images:
         - number_1:
            - num_1.png
            - image_name.png
            - ...
         - number_2:
            - ...
         - ...


##### `create_test_train_val.py`
This seperates your data using symlinks into train, test, and val sets in a 80:10:10 split respectively. These can be edited
in `config.py`. The script will create folders `train`, `test`, and `val` if already not created in the root directory of your
dataset. The mnist example would look like this. Run an `ls -l` on a file in one of the folders to see that it symlinks to the correct file.


data_dir:
   - mnist:
      - README.txt
      - train:
         - number_1
            - image.png
         - number_2:
            - image.png
            - ...
      - test:
         - same as train
      - val:
         - same as test and train
      - images:
         - number_1:
            - num_1.png
            - image_name.png
            - ...
         - number_2:
            - ...
         - ...


##### `createBirdRecords.py`

Tensorflow expects its data in the form of `TFRecords`. This script will go through the `train`, `test`, and `val` folders
and create a TF Record for each *folder*. The resulting records will be stored in the `records` directory in `data_dir` as
`train.tfrecord`, `test.tfrecord`, and `val.tfrecord` respectively.







