import os
import glob
import cv2

ls_path = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center/**/*.tif")

cv2.namedWindow('temp', cv2.WINDOW_NORMAL)
for path in ls_path:
    img = cv2.imread(str(path))
    cv2.imshow('temp', img)
    key = cv2.waitKey(0)

    if key != 97:
        pass
    elif key == 97:
        os.remove(path)