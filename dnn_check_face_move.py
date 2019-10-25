import cv2
import numpy as np
import dlib
import glob

# load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

def is_face(image):
    
    result_img = image.copy()
    h, w, _ = result_img.shape
    blob = cv2.dnn.blobFromImage(result_img, 1.0, (h, w), [104, 117, 123], False, False)
    net.setInput(blob)

    # inference, find faces
    detections = net.forward()
    # postprocessing

    faceBoxRectangleS = None
    accuracy = 0.

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            accuracy = confidence * 100
            faceBoxRectangleS =  dlib.rectangle(left=x1,top=y1,right=x2, bottom=y2)

    if faceBoxRectangleS == None:
        return False

    if accuracy < 90:
        return False

    return True




import shutil
import os
def move_error(error_list):
    for x_dir in error_list: #얼굴이 없는 이미지의 경로를 하나씩 갖고온다
        # print()
        # print(x_dir)

        img_name = x_dir.replace(folder_path + x_path + '\\' , '') 
        #파일의 이름만 갖고오기
        # print(img_name)
        y_dir = folder_path + y_path + '\\' +  img_name
        #같은 이름의 파일의 Y 경로를 저장한다
        dest_dir = folder_path + "error_"

        if not os.path.isdir(dest_dir + x_path):
                os.makedirs(dest_dir + x_path)
        if not os.path.isdir(dest_dir + y_path):
                os.makedirs(dest_dir + y_path)
        shutil.move(x_dir, dest_dir + x_path + '\\' + img_name)
        shutil.move(y_dir, dest_dir + y_path + '\\' + img_name)

folder_path = "D:\Data\TrashData\\"
x_path = 'X'
y_path = 'Y'
glob_path = folder_path + x_path + '/*jpg'
filenames = glob.glob(glob_path)


if len(filenames) == 0:
    print("no such directory")

error_list = []
for img_fp in filenames:
    # print()
    # print(img_fp)
    image = cv2.imread(img_fp)
    if is_face(image):
        continue
    
    error_list.append(img_fp)


move_error(error_list)

        

    





    

