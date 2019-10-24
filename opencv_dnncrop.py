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

def crop_progress(image):

    result_img = image.copy()
    h, w, _ = result_img.shape
    blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)

    # inference, find faces
    detections = net.forward()
    # postprocessing
    if detections.shape[2] == 0:
        print('empty')
    faceBoxRectangleS = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            print(x1, y1, x2, y2)
            faceBoxRectangleS =  dlib.rectangle(left=x1,top=y1,right=x2, bottom=y2)

    if faceBoxRectangleS == None:
        print("empty")
        return None
            # draw rects

            # cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
            
    # # inference time

    # # visualize



    # if len(rects) == 0:
    #     # print("no face points found")
    #     return

    # for rect in rects:
    #     offset = 0
    #     top = rect.top()
    #     bottom = rect.bottom() - 0
    #     left = rect.left() + offset
    #     right = rect.right() - offset



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
    
    return cropped_image

def save_img(image, path, location):
    suffix = get_suffix(path) #suffix = '.jpg'
    image_name = path.replace(folder_path+'\\', '')
    image_name = image_name.replace(suffix, '')
    wfp_crop = location + '/{}_crop.jpg'.format(image_name)

    cv2.imwrite(wfp_crop, image)
    # print('Dump to {}'.format(wfp_csrop))

def main(save_path):

    glob_path = folder_path + '/*jpg'

    filenames = glob.glob(glob_path)

    if len(filenames) == 0:
        print("no such directory")

    for img_fp in filenames:
        print()
        print(img_fp)
        img_ori = cv2.imread(img_fp)
        cropped_image = img_ori
        cropped_image = crop_progress(img_ori)
        if cropped_image is None:
            # print('crop none')
            continue
        # cv2.imshow("cropped", cropped_image)
       
        save_img(cropped_image, img_fp, './' + save_path)
        ##################### 크롭 이미지 출력
        
        




        
folder_path = 'D:\Sangmin\FaceCrop\\test_korean'
STD_SIZE = 224
main('test_korean_out')
print("finished")