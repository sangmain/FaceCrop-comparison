import os
from glob import glob



folder_path = 'D:\\Data\\FaceData\\KR_FACE_GATHERED'
second_path = 'D:\\Data\\224\\36000'

filenames_ori = glob(folder_path +'/*jpg')
filenames_crop = glob(second_path +'/*jpg')


import ntpath
for i in range(len(filenames_ori)):
    filenames_ori[i] = ntpath.basename(filenames_ori[i])

for i in range(len(filenames_crop)):
    filenames_crop[i] = ntpath.basename(filenames_crop[i])

print(filenames_ori[0])
print(filenames_crop[0])

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

print(len(filenames_ori))
print(len(filenames_crop))
error = Diff(filenames_ori, filenames_crop)
print(len(error))

with open('./notfound.txt', 'w') as f:
    for fname in error:
        f.write(fname + '\n')
