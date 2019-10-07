import numpy as np
import cv2
import dlib
# import scipy.io as sio
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
# __init__.py makes error disappear
import glob

STD_SIZE = 128

is_gpu = True;
is_find_landmark = True;
is_find_bbox = True;

foldername = 'C:/Data/FaceData/origin_image_sample/*jpg'

def main():
    print("aaaa")
    # 2. load dlib model for face detection and landmark used for face cropping
    if is_find_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if is_find_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    filenames = glob.glob(foldername)

    for img_fp in filenames:
            
        img_ori = cv2.imread(img_fp)
        if is_find_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        ind = 0
        
        suffix = get_suffix(img_fp) #img_fp 의 절대경로에서 .jpg만 뺸다

        if len(rects) == 0:
            print("no face points found")
            continue

        print('측면')

        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if is_find_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)

            ##################### 크롭 이미지 출력
            if True:
                wfp_crop = './origin/{}_{}_crop.jpg'.format(img_fp.replace(suffix, '')[20:], ind)

                cv2.imwrite(wfp_crop, img)
                print('Dump to {}'.format(wfp_crop))

main()