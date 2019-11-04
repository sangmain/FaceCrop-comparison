import cv2
import os
import dlib

cascade = cv2.CascadeClassifier('D:/Sangmin/BITProjects/lsm/haarcascade_frontalface_alt.xml')
dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    minisize = (gray.shape[1], image.shape[0])
    miniframe = cv2.resize(gray, minisize)

    faces = cascade.detectMultiScale(miniframe)

    faceBoxRectangleS = None
    for i in faces:
        x, y, w, h = [v for v in i]
        faceBoxRectangleS =  dlib.rectangle(left=x,top=y,right=w, bottom=h)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255))

    if faceBoxRectangleS == None:
        suffix = get_suffix(filename) #suffix = '.jpg'
        image_name = filename.replace(folder_path+'\\', '')
        with open('./notfound.txt', 'a+') as f:
            f.write(image_name + '\n')
        print("face not found")
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