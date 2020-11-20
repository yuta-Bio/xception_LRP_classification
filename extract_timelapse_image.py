import os
import glob
import numpy as np
import cv2

path = (":///home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/201013_timelapse_cropped")
dst_path = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/from_timelapse_to_image/images")
ls_path = glob.glob(str(path) + "/*.mp4")
print(len(ls_path))
for i_num, path in enumerate(ls_path):
    print("total " + str(len(ls_path)) + " current " + str(i_num) + " " + str(path), end = "\r")
    movie = cv2.VideoCapture(str(path))
    if movie.isOpened() is False:
        exit()

    total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    # flg, img = movie.read()
    img = np.empty(0)
    for j_num in range(int(total_frame)):
        flg, img = movie.read()
        cv2.imwrite(str(os.path.join(dst_path, os.path.basename(path)[:-4] + "_" + str(j_num) + ".png")), img)
    movie.release()
