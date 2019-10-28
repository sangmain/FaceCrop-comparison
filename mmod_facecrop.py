import cv2
import dlib
import argparse
import time
import numpy as np
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
import glob
import dlib
import sys
import multiprocessing as mp

STD_SIZE = 224
dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

face_regressor = dlib.shape_predictor(dlib_landmark_model)
# handle command line arguments
folder_path = 'C:\Data\korean_face'

glob_path = folder_path + '/*jpg'

filenames = glob.glob(glob_path)


def save_img(image, path, location):
    suffix = get_suffix(path) #suffix = '.jpg'
    image_name = path.replace(folder_path+'\\', '')
    image_name = image_name.replace(suffix, '')
    wfp_crop = location + '/{}_crop.jpg'.format(image_name)

    cv2.imwrite(wfp_crop, image)
    
n = 0

def process2(image):


        if image is None:
            print("Could not read input image")
            return False


        start = time.time()

        # apply face detection (cnn)
        faces_cnn = cnn_face_detector(image, 1)

        end = time.time()
        print("CNN : ", format(end - start, '.2f'))

        # loop over detected faces
        for face in faces_cnn:
            left = face.rect.left()
            top = face.rect.top()
            right = face.rect.right()
            bottom = face.rect.bottom()

            faceBoxRectangleS =  dlib.rectangle(left=left,top=top,right=right, bottom=bottom)

            # - use landmark for cropping
            pts = face_regressor(image, faceBoxRectangleS).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = parse_roi_box_from_landmark(pts)

            height, width, _ =  image.shape

            ########## left
            if roi_box[0] < 0:
                roi_box[0] = 0
            ########## right        
            if roi_box[1] < 0:
                roi_box[1] = 0
            ########## width
            if roi_box[2] > width:
                roi_box[2] = width
            ########## height
            if roi_box[3] > height:
                roi_box[3] = height


            cropped_image = crop_img(image, roi_box)

            # forward: one step
            
            cropped_image = cv2.resize(cropped_image, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            


            print(cropped_image.shape)
            # image = cropped_image
            # path = filename
            # location= 'D:\Sangmin\FaceCrop\\test_crop_out'
            # suffix = get_suffix(path) #suffix = '.jpg'
            # image_name = path.replace(folder_path+'\\', '')
            # image_name = image_name.replace(suffix, '')
            # wfp_crop = location + '/{}_crop.jpg'.format(image_name)
            # save_img(cropped_image, filename, './test_crop_out')
            
            return cropped_image
            # cv2.imwrite(wfp_crop, image)
            print('saved')
        return False


if __name__ == '__main__':
        # pool = mp.Pool(processes=mp.cpu_count())


        results = []
        for filename in filenames:  
        # load input image
            image = cv2.imread(filename)
            # result = pool.apply_async(process2, (image,), callback=None)
            # results.append(result)
            # cv2.imshow(result.get(1))
            # cv2.waitKey(0)
            image = process2(image)
            save_img(image, filename, 'D:\Sangmin\FaceCrop\\new_crop')
            # cv2.imshow("aa", image)
            # cv2.waitKey(0)

        # for r in results:
        #     r.wait()
