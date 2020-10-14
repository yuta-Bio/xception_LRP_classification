import cv2
import numpy as np
import keras
from keras import models
import matplotlib.pyplot as plt
from manipulate_img import img_to_square


movie_path = ('/home/pmb-mju/DL_train_data/archive/200723_lateral_root_timelapse_cafe_dr5/Position048_chan00.mp4')
movie = cv2.VideoCapture(str(movie_path))
shape = (224, 224)
if movie.isOpened():
    exit()

total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)

model_path = ('hoge')
resnet = models.load_model(model_path, compile=False)
ls_stage = []
for num in range(total_frame):
    flg, img = movie.read()
    img = img_to_square(img)
    #crop image
    crop_rate = 3
    ht = img.shape[0]
    wd = img.shape[1]
    img = img[ht//crop_rate: ht - (ht//crop_rate), wd // crop_rate : wd - (wd // crop_rate)]

    #to square
    if img.shape[0] == img.shape[1]:
        pass
    elif img.shape[0] > img.shape[1]:
        dif = img.shape[0] - img.shape[1]
        img = np.delete(img, np.s_[-(dif//2+1):], 0)
        img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 0)
    else:
        dif = img.shape[1]-img.shape[0]
        img = np.delete(img, np.s_[-(dif//2+1):], 1)
        img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 1)

    img = cv2.resize(img, shape)
    img = img.astype('float') / 255
    img = np.reshape(img, (1, shape[0], shape[1], 1))
    inf = resnet.predict(img)
    lrp_stage = inf.argmax()
    ls_stage.append(lrp_stage)
