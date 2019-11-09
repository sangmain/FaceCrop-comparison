import os
from glob import glob



second_path = 'D:\\Data\\New_data\\X_train'

filenames_crop = glob(second_path +'/*jpg')


import ntpath

for fname in filenames_crop:
    os.rename(fname, fname[:-3])


