import cv2
import numpy as np
import dlib
# import scipy.io as sio
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
# __init__.py makes error disappear
import glob

# load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

img = cv2.imread('D:\Sangmin\FaceCrop\\test_korean\\17080801_S001_L02_E01_C1.jpg')
result_img = img.copy()
h, w, _ = result_img.shape
blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
net.setInput(blob)

# inference, find faces
detections = net.forward()

# postprocessing
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        # draw rects
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)

# inference time

# visualize
cv2.imshow('result', result_img)
cv2.waitKey(0)