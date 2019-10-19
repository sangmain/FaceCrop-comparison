import cv2
import dlib
import argparse
import time
import glob
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark

# handle command line arguments

# loop over detected faces
folder_path = 'D:\Sangmin\FaceCrop\img'

glob_path = folder_path + '/*jpg'

filenames = glob.glob(glob_path)


def save_img(image, path, location):
    suffix = get_suffix(path) #suffix = '.jpg'
    image_name = path.replace(folder_path+'\\', '')
    image_name = image_name.replace(suffix, '')
    wfp_crop = location + '/{}_crop.jpg'.format(image_name)

    cv2.imwrite(wfp_crop, image)


cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

for filename in filenames:
    image = cv2.imread(filename)
    start = time.time()

    faces_cnn = cnn_face_detector(image, 1)

    end = time.time()
    print("CNN : ", format(end - start, '.2f'))

    # loop over detected faces
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

        # draw box over face
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
        print("left", x, "top", y, "right", w, "bottom", h)


        # display output image
        save_img(image, filename, './Taehwan')
        print('saved')
