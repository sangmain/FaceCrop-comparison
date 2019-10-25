import cv2
import os

def facecrop(image_path):
    cascade = cv2.CascadeClassifier('D:/Sangmin/BITProjects/lsm/haarcascade_frontalface_alt.xml')


    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    ori_img = cv2.imread(image_path)

    print(image.shape)

    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)

    faces = cascade.detectMultiScale(miniframe)

    print(faces)
    for i in faces:
        x, y, w, h = [v for v in i]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255))

        find_face = ori_img[y : y + h, x : x + w]
        find_face = cv2.resize(find_face, (128, 128))

        directory, file_name = os.path.split(image_path)
        cv2.imwrite(save_path + file_name, find_face)
        print('saved')

    return


# image_path = 'C:/Data/FaceData/origin_image_sample/1-02.jpg'
save_path = 'D:/cropped_image/'
import glob
if __name__ == '__main__':
    images = glob.glob('F:/kface_sample/*.jpg')

    for fname in images:
        facecrop(fname)