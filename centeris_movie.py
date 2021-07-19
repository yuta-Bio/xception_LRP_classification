import glob
import os
import copy
import numpy as np
import cv2
import re
import sys

class img_data:
    def __init__(self, img):
        self.img = img
        self.move_x = 0
        self.move_y = 0

    def translate(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mid_x = self.img.shape[1] // 2
            mid_y = self.img.shape[0] // 2
            self.move_x = mid_x - x
            self.move_y = mid_y - y
            self.img = cv2.warpAffine(self.img, np.float32([[1, 0, self.move_x], [0, 1, self.move_y]]), (self.img.shape[1], self.img.shape[0]))

args = sys.argv


path_src = args[1]
ls_path = glob.glob(path_src + "/**/*.avi", recursive=True)
dst_dir = r"C:\Users\PMB_MJU\timelapse\resource\centered"
dir_dst = r"C:\Users\PMB_MJU\x40_images_center_plus"
for i_num, path in enumerate(ls_path):
    movie = cv2.VideoCapture(str(path))
    if movie.isOpened() is False:
        exit()

    # Define the codec and create VideoWriter object

    total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    # flg, img = movie.read()
    img = np.empty(0)
    for j_num in range(int(total_frame - 1)):
        flg, img = movie.read()
    movie.release()
    movie = cv2.VideoCapture(str(path))
    image_data = img_data(img)

    wd = img.shape[1]
    hg = img.shape[0]
    cv2.namedWindow("image translate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image translate", image_data.translate)

    flg_key = 0
    while(True):
        show_img = copy.copy(image_data.img)
            # add crossed line in showing image
        show_img = copy.copy(image_data.img)
        width = show_img.shape[1]
        height = show_img.shape[0]
        #horizontal line
        show_img = cv2.line(show_img, (width // 2, 0), (width // 2, height), (255, 255, 255), 3)

        #vertical line
        show_img = cv2.line(show_img, (0, height // 2), (width, height // 2), (255, 255, 255), 3)
        cv2.imshow("image translate", show_img)
        flg_key = cv2.waitKey(1)
        if flg_key == 97 or flg_key == 98:  #97 == a key
            # cv2.imwrite(str(path), image_data.img)
            break
    if flg_key == 98:
        continue
    for j_num in range(int(total_frame)):
        flg, img = movie.read()
        img = cv2.warpAffine(img, np.float32([[1, 0, image_data.move_x], [0, 1, image_data.move_y]]), (img.shape[1], img.shape[0]))
        dir_dst_img = os.path.join(dir_dst, str(i_num) + "_" + str(j_num) + ".png")
        if j_num < 30 and j_num % 2 == 0:
            cv2.imwrite(dir_dst_img, img)
        elif j_num % 6 == 0:
            cv2.imwrite(dir_dst_img, img)
