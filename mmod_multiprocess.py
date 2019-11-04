
import cv2
import dlib
import glob
import mmod_facecrop as md

import multiprocessing as mp

STD_SIZE = 224

folder_path = 'D:\Data\FaceData\origin_image_sample'
save_path = 'D:\Sangmin\FaceCrop\origin'

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
        
        result = pool.apply_async(md.process2, (image, filename, folder_path, save_path), callback= aaa)
        results.append(result)

        # print(type(result.get(timeout=1)))
        # cv2.imshow(result.get(timeout=1))
        # cv2.waitKey(0)
    # print(string)

    # pool.close()
    # pool.join()



    for r in results:
        r.wait()