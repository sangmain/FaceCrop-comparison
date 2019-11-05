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
import ntpath

dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

face_regressor = dlib.shape_predictor(dlib_landmark_model)


def save_img(image, filename, save_path):
    image_name = ntpath.basename(filename)
    wfp_crop = save_path + '/{}'.format(image_name)
    print(wfp_crop)
    cv2.imwrite(wfp_crop, image)
    

def crop_process(image, filename, folder_path, save_path, size=224):

    if image is None:
        print("Could not read input image")
        return False


    start = time.time()

    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(image, 1)

    end = time.time()
    print("CNN : ", format(end - start, '.2f'))
    print(len(faces_cnn))
    if len(faces_cnn) == 0:
        print("no face found")
        return False    

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
        cropped_image = cv2.resize(cropped_image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        save_img(cropped_image, filename, save_path)
        print('saved')        
        return cropped_image
        
    return False
