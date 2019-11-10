import numpy as np
import cv2
import dlib
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
import glob

STD_SIZE = 128


############# load dlib model for face detection and landmark used for face cropping
dlib_landmark_model = './models/shape_predictor_68_face_landmarks.dat'
cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

face_regressor = dlib.shape_predictor(dlib_landmark_model)
face_detector = dlib.get_frontal_face_detector()

def crop_progress(image):

    rects = face_detector(image, 1)

    if len(rects) == 0: ## frontal face detector가 얼굴을 찾았는가?
        print('using cnn')
        rects = cnn_face_detector(image, 1)
    
        if len(rects) == 0: ### mmod가 얼굴을 찾았는가?
            print('could not find face.')
            return

        rect = rects[0].rect ###rect를 mmod가 찾은 것으로 진행한다
        
    else:
        rect = rects[0] #### frontal face detector가 찾은 얼굴로 진행한다

    top = rect.top()
    bottom = rect.bottom()
    left = rect.left()
    right = rect.right()


    faceBoxRectangleS =  dlib.rectangle(left=left,top=top,right=right, bottom=bottom)

    # - use landmark for cropping
    pts = face_regressor(image, faceBoxRectangleS).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T
    roi_box = parse_roi_box_from_landmark(pts)


    cropped_image = crop_img(image, roi_box)

    # forward: one step
    cropped_image = cv2.resize(cropped_image, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    
    return cropped_image

if __name__ == '__main__':
    image = cv2.imread('D:\Data\\17081601_S001_L06_E01_C18.jpg')
    result = crop_progress(image)

    cv2.imshow("aa", result)
    cv2.waitKey(0)