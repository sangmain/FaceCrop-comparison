
import cv2
import dlib
import glob
import multiprocessing as mp

import mmod_facecrop as mmod
import opencv_dnncrop as dnn
import frontal_facecrop as frt

STD_SIZE = 224

folder_path = 'D:\\Data\\FaceData\\KR_FACE_GATHERED'
save_path = 'D:\\Data\\224\\36000'

glob_path = folder_path + '/*jpg'
filenames = glob.glob(glob_path)

string = []
def aaa(result):
    string.append('aaa')

if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())

    results = []
    for filename in filenames:  
    # load input image
        image = cv2.imread(filename)
        # cv2.imshow("aa", image)
        # cv2.waitKey(0)
        
        ################ MMOD ################
        # result = pool.apply_async(mmod.crop_process, (image, filename, folder_path, save_path), callback= aaa)
        # results.append(result)

        ################ DNN ################
        result = pool.apply_async(dnn.crop_process, (image, filename, folder_path, save_path), callback= aaa)
        results.append(result)

        ################ Frontal Face ################
        result = pool.apply_async(frt.crop_process, (image, filename, folder_path, save_path), callback= aaa)
        results.append(result)
    # pool.close()
    # pool.join()



    for r in results:
        r.wait()