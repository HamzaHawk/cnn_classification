import os
import sys

"""

Script that will generate labels for images

input: root folder containing sub directories. Each subdirectory contains images of a certain label.

The label is the name of the sub directory

root -
   - birds
   - cars
   - ...

This file is taken in by build_image_data.py which creates tensorflow records

"""

root = sys.argv[1]
label_file = sys.argv[2]

"""
with open(label_file, "a") as lf:
   for root, dirs, files in os.walk(root):
      for dd in dirs:
         for r, d, img in os.walk(root+dd):
            label = dd
            for i in img:
               i = root + dd + "/" + i
               ii = os.path.splitext(i)[1]
               if ii == ".csv":
                  continue
               lf.write(i+"|"+label+"\n")
"""

with open(label_file, "a") as lf:
   for root, dirs, files in os.walk(root):
      for d in dirs:
         lf.write(d+"\n")

