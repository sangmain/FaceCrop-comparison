import cv2
import dlib
import argparse
import time
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark
import glob
STD_SIZE = 224
# handle command line arguments
folder_path = 'D:\Data\korean_face'

glob_path = folder_path + '/*jpg'

filenames = glob.glob(glob_path)


def save_img(image, path, location):
    suffix = get_suffix(path) #suffix = '.jpg'
    image_name = path.replace(folder_path+'\\', '')
    image_name = image_name.replace(suffix, '')
    wfp_crop = location + '/{}_crop.jpg'.format(image_name)

    cv2.imwrite(wfp_crop, image)


cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

for filename in filenames:
# load input image
    image = cv2.imread(filename)

    if image is None:
        print("Could not read input image")
        exit()

    # initialize hog + svm based face detector
    # hog_face_detector = dlib.get_frontal_face_detector()

    # initialize cnn based face detector with the weights

    # start = time.time()

    # # apply face detection (hog)
    # faces_hog = hog_face_detector(image, 1)

    # end = time.time()
    # print("Execution Time (in seconds) :")
    # print("HOG : ", format(end - start, '.2f'))

    # loop over detected faces
    # for face in faces_hog:
    #     x = face.left()
    #     y = face.top()
    #     w = face.right() - x
    #     h = face.bottom() - y

    #     # draw box over face
    #     cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)


    start = time.time()

    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(image, 1)

    end = time.time()
    print("CNN : ", format(end - start, '.2f'))

    # loop over detected faces
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right()
        h = face.rect.bottom()

        #  draw box over face
        # cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

        print("left", x, "top", y, "right", w, "bottom", h)

    # write at the top left corner of the image
    # for color identification

        # cropped_image = image[int(x):int(w), int(y):int(h)]
        cropped_image = image[y:h, x:w]
        print(cropped_image.shape)
        # cv2.imshow("cropped", cropped_image)
        # cv2.waitKey(0)
        cropped_image = cv2.resize(cropped_image, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)

        save_img(cropped_image, filename, './mmod_crop')
        print('saved')
