import os
import datetime
import shutil
import glob
import keras
from keras import applications, layers, Model
import numpy as np
import cv2

base_dir = ('/home/pmb-mju/dl_result')
image_path = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus")
basename = datetime.datetime.now().strftime("%y%m%d%H%M")
path = os.path.join(base_dir, basename)
# os.mkdir(path)
shutil.copytree(image_path, path)
shutil.copyfile(__file__, str(os.path.join(path, os.path.basename(__file__))))
ls_path = glob.glob(path + "/**/*.tif", recursive=True)

# create model
shape = (500, 500, 3)
base_model = applications.ResNet50V2(include_top = False, input_shape = shape)
x = layers.Flatten()(base_model.output)
output = layers.Dense(1)(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights("/home/pmb-mju/dl_result/2011251614/LRP_classifier_best.h5")
crop_rate = 3.5
for num, path in enumerate(ls_path):
    print("total is " + str(len(ls_path)) + " : current is " + str(num) + str(path), end = "\r")
    src_img = cv2.imread(str(path))

    # image sub process
    ht = src_img.shape[0]
    wd = src_img.shape[1]
    src_img = src_img[int(ht/crop_rate): int(ht - (ht/crop_rate)), int(wd / crop_rate) : int(wd - (wd / crop_rate))]
    #reshape image to square
    if src_img.shape[0] == src_img.shape[1]:
        pass
    elif src_img.shape[0] > src_img.shape[1]:
        dif = src_img.shape[0]-src_img.shape[1]
        src_img = np.delete(src_img, np.s_[-(dif//2+1):], 0)
        temp_img = np.delete(src_img, np.s_[:abs(dif-(dif//2)-1)], 0)
    else:
        dif = src_img.shape[1]-src_img.shape[0]
        src_img = np.delete(src_img, np.s_[-(dif//2+1):], 1)
        src_img = np.delete(src_img, np.s_[:abs(dif-(dif//2)-1)], 1)
    src_img = cv2.resize(src_img, shape[:2])

    pre_img = np.reshape(src_img, (1, 500, 500, 3))
    pre_img = pre_img.astype('float32') / 255
    stage = model.predict(pre_img)
    stage = stage[0] * 8
    dst_img = cv2.putText(src_img, "stage : " + str(stage), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(str(path), dst_img)
