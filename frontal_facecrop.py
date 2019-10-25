import numpy as np
import cv2
import dlib
# import scipy.io as sio
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
# __init__.py makes error disappear
import glob




# folder_path = 'C:\Data\FaceData\origin_image_sample'

############# load dlib model for face detection and landmark used for face cropping
dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)
face_detector = dlib.get_frontal_face_detector()

def crop_progress(image):

    rects = face_detector(image, 1)


    if len(rects) == 0:
        # print("no face points found")
        return

    for rect in rects:
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
        img_ori = cv2.imread(img_fp, )
        cropped_image = img_ori
        cropped_image = crop_progress(img_ori)
        if cropped_image is None:
            # print('crop none')
            continue
        # cv2.imshow("cropped", cropped_image)
       
        save_img(cropped_image, img_fp, save_path)
        ##################### 크롭 이미지 출력
        
        




        
folder_path = 'D:\Data\FaceData\Predict_data'
STD_SIZE = 224
main('D:\Data\FaceData\\test_data')
print("finished")