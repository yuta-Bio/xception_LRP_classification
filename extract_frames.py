import glob
import os
import copy
import numpy as np
import cv2

path_src = "/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/210112_dr5_cafe_timelapse/Position025_chan00.mp4"
# path_dst = "/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/timelapse/resource/dst/201229_dr5_cafe/Position009_chan00_croped.mp4"

movie = cv2.VideoCapture(str(path_src))
if movie.isOpened() is False:
    exit()

total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
ht = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
wd = movie.get(cv2.CAP_PROP_FRAME_WIDTH)

flg_key = 0

# fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
# out = cv2.VideoWriter(str(dst_path),fourcc, 6.0, (500,500))

crop_rate = 3.5
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
for num in range(int(total_frame)):
    flg, img = movie.read()
    cv2.putText(img, str(num + 1) + " / "+ str(total_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 3)
    cv2.imshow("frame", img)
    key = cv2.waitKey()
    if key == ord("a"):
        break
cv2.destroyAllWindows()
movie.release()

# out.write(img)
# out.release()
