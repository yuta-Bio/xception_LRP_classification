import glob
import os
import copy
import numpy as np
import cv2

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

ls_path = glob.glob("temp/**/*.mp4")
shape = (500, 500)

for num, path in enumerate(ls_path):
    movie = cv2.VideoCapture(str(path))
    if movie.isOpened() is False:
        exit()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(path)[:-4] + ".avi",fourcc, 6.0, shape)

    total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    flg, img = movie.read()
    image_data = img_data(img)

    wd = img.shape[1]
    hg = img.shape[0]
    cv2.namedWindow("image translate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image translate", image_data.translate)

    while(True):
        show_img = copy.copy(image_data.img)
        cv2.imshow("image translate", show_img)
        if cv2.waitKey(1) == 97:  #97 == a key
            cv2.imwrite(str(path), image_data.img)
            break

    img = image_data.img

    # crop image
    img = img[hg //2 -shape[0] // 2 : hg // 2 + shape[0] // 2, wd // 2 - shape[1] //2 : wd // 2 + shape[1] // 2]
    out.write(img)
    for num in range(total_frame - 1):
        flg, img = movie.read()
        img = cv2.warpAffine(img, np.float32([[1, 0, image_data.move_x], [0, 1, image_data.move_y]]), (img.shape[1], img.shape[0]))
        img = img[hg //2 -shape[0] // 2 : hg // 2 + shape[0] // 2, wd // 2 - shape[1] //2 : wd // 2 + shape[1] // 2]
        out.write(img)
    out.release()
