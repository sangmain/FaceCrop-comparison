from glob import glob
import os
import shutil

folder_path = 'D:\\Data\\FaceData\\aihub\\High_Resolution_2018\\High_Resolution_2018\\All'
save_path = 'D:\\Data\\FaceData\\KR_FACE_GATHERED'

folder_names = next(os.walk(folder_path))[1]

for i in range(len(folder_names)): # number of files
    for j in range(1, 7): # number of lights
        file_path = os.path.join(folder_path, folder_names[i], 'S001', 'L0'+ str(j), 'E01')
        filenames = glob(file_path +'/*jpg')
        
        for fname in filenames:
            shutil.copy2(fname, save_path)
