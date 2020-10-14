import os
import glob
import numpy
import cv2

ls_path = glob.glob('/home/pmb-mju/DL_train_data/complete_translate_crop/**/*.tif')
crop_rate = 4

for num, path in enumerate(ls_path):
    print(str(len(ls_path)), str(num), end = '\r')
    img = cv2.imread(str(path))
    # crop image
    ht = img.shape[0]
    wd = img.shape[1]
    img = img[int(ht/crop_rate): int(ht - (ht/crop_rate)), int(wd / crop_rate) : int(wd - (wd / crop_rate))]
    # cv2.imwrite(str(path), img)

cv2.destroyAllWindows()
