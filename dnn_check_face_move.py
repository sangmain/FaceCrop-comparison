import cv2
import numpy as np
import dlib
import glob
import sys

# load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7
# 얼굴인가
def is_face(image, threshold= 90):
    
    result_img = image.copy() #데이터의 복사본 제작
    h, w, _ = result_img.shape #데이터의 크기를 가져온다

    blob = cv2.dnn.blobFromImage(result_img, 1.0, (h, w), [104, 117, 123], False, False)
    net.setInput(blob)

    # inference, find faces
    detections = net.forward()

    accuracy = 0. #예측 결과, 얼굴일 확률을 백분율로 저장한다

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            accuracy = confidence * 100



    if accuracy < threshold: # 얼굴일 확률이 threshold(기본값: 90) 보다 낮다면 얼굴이 아닌것으로 판단
        return False

    return True




import shutil
import os

def move_error(error_list):
    for x_dir in error_list: #얼굴이 없는 이미지의 절대경로를 하나씩 갖고온다
        # print()
        # print(x_dir)

        #파일의 이름만 갖고오기 image.jpg
        img_name = x_dir.replace(folder_path + x_path + '\\' , '') 

        #같은 이름의 파일의 Y 경로를 지정한다
        y_dir = folder_path + y_path + '\\' +  img_name

        # 저장할 위치의 경로를 지정한다
        dest_dir = folder_path + "error_"

        # 저장할 위치가 없으면 만든다
        if not os.path.isdir(dest_dir + x_path):
                os.makedirs(dest_dir + x_path)
        if not os.path.isdir(dest_dir + y_path):
                os.makedirs(dest_dir + y_path)

        #얼굴이 아닌 이미지를 옮긴다
        shutil.move(x_dir, dest_dir + x_path + '\\' + img_name)
        shutil.move(y_dir, dest_dir + y_path + '\\' + img_name)




# 얼굴인지를 확인할 데이터의 위치  "D:\Data\TrashData\\"
folder_path = "D:\Data\TrashData\\"
# 얼굴인지를 확인할 데이터의 위치 내의 X와 Y 파일 이름
x_path = 'X'
y_path = 'Y'

#경로 내에 있는 이미지의 위치를 정대 경로로 가져온다
glob_path = folder_path + x_path + '/*jpg'
filenames = glob.glob(glob_path)


# 찾은 이미지의 갯수가 0개라면 
if len(filenames) == 0:
    print("no such directory")
    sys.exit()

error_list = []
for img_fp in filenames:
    # print()
    # print(img_fp)
    image = cv2.imread(img_fp) #이미지를 절대 경로를 이용해 가져온다

    if is_face(image): # 얼굴인지를 검사하고, 얼굴이면 넘어간다
        continue
    
    
    error_list.append(img_fp) #얼굴이 아니면 에러 목록에 추가


# 에러로 의심되는 이미지들을 지정된 파일 밖으로 옮긴다
move_error(error_list)

        

    





    

