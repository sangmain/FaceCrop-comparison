
import cv2
import dlib
from glob import glob
import multiprocessing as mp
import os

import mmod_facecrop as mmod
import opencv_dnncrop as dnn
import frontal_facecrop as frt

STD_SIZE = 128

folder_path = 'D:\\Sangmin\\FaceCrop\\image'
save_path = 'D:\\Sangmin\\FaceCrop\\out'

glob_path = folder_path + '/*jpg'
filenames = glob(glob_path)

# filenames = []

string = []
def aaa(result):
    string.append('aaa')

if __name__ == '__main__':
    # with open('./notfound.txt', 'r') as f:
    #     filenames = f.readlines()

    pool = mp.Pool(processes=mp.cpu_count())

    results = []
    for filename in filenames:  
        # load input image
        # filename = filename.strip()     
        image = cv2.imread(os.path.join(folder_path,filename))
        # print(filename)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        ################ MMOD ################
        # result = pool.apply_async(mmod.crop_process, (image, filename, folder_path, save_path), callback= aaa)
        # results.append(result)

        ################ DNN ################
        # result = pool.apply_async(dnn.crop_process, (image, filename, folder_path, save_path), callback= aaa)
        # results.append(result)

        ################ Frontal Face ################
        result = pool.apply_async(frt.crop_process, (image, filename, folder_path, save_path), callback= aaa)
        results.append(result)
    # pool.close()
    # pool.join()



    for r in results:
        r.wait()