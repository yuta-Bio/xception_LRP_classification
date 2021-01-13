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

ls_path = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/210112_dr5_cafe_timelapse/*.avi")
dst_dir_path = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/timelapse/resource/dst/210112_dr5_cafe")

for num, path in enumerate(ls_path):
    movie = cv2.VideoCapture(str(path))
    if movie.isOpened() is False:
        exit()

    # Define the codec and create VideoWriter object
    dst_path = os.path.join(dst_dir_path, str(os.path.basename(path))[:-4] + "_croped.mp4")

    total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    # flg, img = movie.read()
    img = np.empty(0)
    for num in range(int(total_frame - 1)):
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
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    out = cv2.VideoWriter(str(dst_path),fourcc, 6.0, (500,500))
    crop_rate = 3.5
    for num in range(166):
        flg, img = movie.read()
        img = cv2.warpAffine(img, np.float32([[1, 0, image_data.move_x], [0, 1, image_data.move_y]]), (img.shape[1], img.shape[0]))
        path_img_dst = str(os.path.join(dst_dir_path, str(os.path.basename(path)) + str(num) + 'centered.png'))
        # cv2.imwrite(path_img_dst, img)
        # img = img[hg //2 -shape[0] // 2 : hg // 2 + shape[0] // 2, wd // 2 - shape[1] //2 : wd // 2 + shape[1] // 2]
        ht = img.shape[0]
        wd = img.shape[1]
        img = img[int(ht/crop_rate): int(ht - (ht/crop_rate)), int(wd / crop_rate) : int(wd - (wd / crop_rate))]
        img = cv2.resize(img, (500, 500))

        out.write(img)
    out.release()
    movie.release()
