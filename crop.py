import os
import glob
import numpy
import cv2

ls_path = glob.glob('/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_5th/images/*.tif')
crop_rate = 3.5

for num, path in enumerate(ls_path):
    print(str(len(ls_path)), str(num), end = '\r')
    img = cv2.imread(str(path))
    # crop image
    ht = img.shape[0]
    wd = img.shape[1]
    img = img[int(ht/crop_rate): int(ht - (ht/crop_rate)), int(wd / crop_rate) : int(wd - (wd / crop_rate))]
    cv2.imwrite(str(path), img)
