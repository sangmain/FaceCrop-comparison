import cv2
import glob
from utils.inference import get_suffix, crop_img, parse_roi_box_from_landmark

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
        img_ori = cv2.imread(img_fp)
        resized_img = cv2.resize(img_ori, (STD_SIZE, STD_SIZE))
        if resized_img is None:
            print('crop none')
            continue
        # cv2.imshow("cropped", cropped_image)

        save_img(resized_img, img_fp, save_path)
        ##################### 크롭 이미지 출력
        
        

        
folder_path = 'D:\Data\Korean 224X224X3 filtering\Y_train'
STD_SIZE = 128
main('D:\Data\Korean 128X128X3 filtering\Y_train')
print("finished")