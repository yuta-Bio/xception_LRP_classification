import numpy as np
import cv2

path = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images/stage3/200923_dr5_ms_x40_ap6_int84_ex4_Image072_ch00.tif")

img = cv2.imread(str(path))
crop_rate = 3.5
ht = img.shape[0]
wd = img.shape[1]
img = img[int(ht/crop_rate): int(ht - (ht/crop_rate)), int(wd / crop_rate) : int(wd - (wd / crop_rate))]

# crop image to square
if img.shape[0] == img.shape[1]:
    pass
elif img.shape[0] > img.shape[1]:
    dif = img.shape[0] - img.shape[1]
    self.pre_src_img = np.delete(img, np.s_[-(dif//2+1):], 0)
    temp_img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 0)
else:
    dif = img.shape[1] - img.shape[0]
    img = np.delete(img, np.s_[-(dif//2+1):], 1)
    img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 1)

print(img.shape,)
img = cv2.resize(img, (128, 128))

cv2.namedWindow('temp', cv2.WINDOW_NORMAL)
cv2.imshow('temp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()