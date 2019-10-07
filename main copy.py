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

folder_path = 'C:\Data\korean_face'

# folder_path = 'C:\Data\FaceData\origin_image_sample'

def main():
    
    # 2. load dlib model for face detection and landmark used for face cropping
    if is_find_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if is_find_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    glob_path = folder_path + '/*jpg'

    filenames = glob.glob(glob_path)

    if len(filenames) == 0:
        print("no such directory")

    cnt = 0
    for img_fp in filenames:
        print(img_fp)
        cnt += 1

        if cnt == 14:
            cnt = 0
        elif cnt > 9:
            continue

        img_ori = cv2.imread(img_fp)
        if is_find_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        ind = 0
        
        suffix = get_suffix(img_fp) #suffix = '.jpg'

        if len(rects) == 0:
            # print("no face points found")
            continue


        for rect in rects:
            offset = 0
            top = rect.top()
            bottom = rect.bottom() - 0
            left = rect.left() + offset
            right = rect.right() - offset


            faceBoxRectangleS =  dlib.rectangle(left=left,top=top,right=right, bottom=bottom)

            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if is_find_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, faceBoxRectangleS).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)

            ##################### 크롭 이미지 출력
            if True:
                pass
                image_name = img_fp.replace(folder_path+'\\', '')
                image_name = image_name.replace(suffix, '')
                wfp_crop = './origin_x/{}_crop.jpg'.format(image_name)

                cv2.imwrite(wfp_crop, img)
                print('Dump to {}'.format(wfp_crop))

main()
print("finished")