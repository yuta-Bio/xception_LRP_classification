import os
import shutil
import random
import glob
import cv2
import numpy as np

src_path = r"C:\Users\PMB_MJU\x40_images_center_plus"
ls_path = glob.glob(src_path + '/images/*.png')
random.shuffle(ls_path)
path_stage0 = os.path.join(src_path, 'stage0')
path_stage1 = os.path.join(src_path, 'stage1')
path_stage2 = os.path.join(src_path, 'stage2')
path_stage3 = os.path.join(src_path, 'stage3')
path_stage4 = os.path.join(src_path, 'stage4')
path_stage5 = os.path.join(src_path, 'stage5')
path_stage6 = os.path.join(src_path, 'stage6')
path_stage7 = os.path.join(src_path, 'stage7')

for path in ls_path:
    img = cv2.imread(path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    key = cv2.waitKey(0)
    if key == 48:
        shutil.move(path, path_stage0)
    elif key == 49:
        shutil.move(path, path_stage1)
    elif key == 50:
        shutil.move(path, path_stage2)
    elif key == 51:
        shutil.move(path, path_stage3)
    elif key == 52:
        shutil.move(path, path_stage4)
    elif key == 53:
        shutil.move(path, path_stage5)
    elif key == 54:
        shutil.move(path, path_stage6)
    elif key == 55:
        shutil.move(path, path_stage7)
cv2.destroyAllWindows()
