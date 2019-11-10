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

dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)

conf_threshold = 0.7


def save_img(image, filename, save_path, folder_path):
    suffix = get_suffix(filename) #suffix = '.jpg'
    image_name = filename.replace(folder_path+'\\', '')
    image_name = image_name.replace(suffix, '')
    wfp_crop = save_path + '/{}.jpg'.format(image_name)
    cv2.imwrite(wfp_crop, image)

def crop_process(image, filename, folder_path, save_path, size=224):

    if image is None:
        print("Could not read input image")
        return False


    h, w, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
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
            #, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            accuracy = confidence * 100
            faceBoxRectangleS =  dlib.rectangle(left=x1,top=y1,right=x2, bottom=y2)

    if faceBoxRectangleS == None:
        suffix = get_suffix(filename) #suffix = '.jpg'
        image_name = filename.replace(folder_path+'\\', '')
        with open('./notfound2.txt', 'a+') as f:
            f.write(image_name + '\n')
        return False

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
    save_img(cropped_image, filename, save_path, folder_path)
    # print('saved')        
    return cropped_image