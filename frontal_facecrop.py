import numpy as np
import cv2
import dlib
# import scipy.io as sio
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
# __init__.py makes error disappear
import glob


############# load dlib model for face detection and landmark used for face cropping
dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)
face_detector = dlib.get_frontal_face_detector()

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

    rects = face_detector(image, 1)


    if len(rects) == 0:
        suffix = get_suffix(filename) #suffix = '.jpg'
        image_name = filename.replace(folder_path+'\\', '')
        with open('./notfound.txt', 'a+') as f:
            f.write(image_name + '\n')
        print("face not found")
        return False

    rect = rects[0]
    offset = 0
    top = rect.top()
    bottom = rect.bottom() - 0
    left = rect.left() + offset
    right = rect.right() - offset


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
    save_img(cropped_image, filename, save_path, folder_path)
    # print('saved')        
    return cropped_image