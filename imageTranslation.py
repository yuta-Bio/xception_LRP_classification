import os
import glob
import copy
import numpy as np
import cv2

class img_data:
    def __init__(self, path, img):
        self.img = img
        self.path = path

    def translate(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mid_x = self.img.shape[1] // 2
            mid_y = self.img.shape[0] // 2
            move_x = mid_x - x
            move_y = mid_y - y
            self.img = cv2.warpAffine(self.img, np.float32([[1, 0, move_x], [0, 1, move_y]]), (self.img.shape[1], self.img.shape[0]))

ls_path = glob.glob("/home/pmb-mju/DL_train_data/complete_translate/**/*.tif")
for path in ls_path:
    img = cv2.imread(str(path))
    image_data = img_data(str(path), img)
    high, width = img.shape[:2]

    cv2.namedWindow("image translate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image translate", image_data.translate)
    while(True):
        show_img = copy.copy(image_data.img)
        #horizontal line
        show_img = cv2.line(show_img, (width // 2, 0), (width // 2, high), (255, 255, 255), 3)

        #vertical line
        show_img = cv2.line(show_img, (0, high // 2), (width, high // 2), (255, 255, 255), 3)

        cv2.imshow("image translate", show_img)
        if cv2.waitKey(1) == 97:  #97 == a key
            cv2.imwrite(str(path), image_data.img)
            break
