import os
import shutil
import random
import glob
import cv2
import numpy as np

src_path = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_4th")
ls_path = glob.glob(src_path + '/images/*.tif')
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
    if key == 49:
        shutil.move(path, path_stage1)
    if key == 50:
        shutil.move(path, path_stage2)
    if key == 51:
        shutil.move(path, path_stage3)
    if key == 52:
        shutil.move(path, path_stage4)
    if key == 53:
        shutil.move(path, path_stage5)
    if key == 54:
        shutil.move(path, path_stage6)
    if key == 55:
        shutil.move(path, path_stage7)
cv2.destroyAllWindows()
