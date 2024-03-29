import os
import datetime
import shutil
import glob
import keras
from keras import applications, layers, Model
import numpy as np
import cv2

ls_path = glob.glob('/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/from_timelapse_to_image_201127/cropped/*.mp4')
dst_dir_path = r"C:\Users\PMB_MJU\timelapse\temp"
# create model
shape = (500, 500, 3)
base_model = applications.ResNet50(include_top = False, input_shape = shape)

# froze model's layer
# for num, layer in enumerate(base_model.layers):
#     base_model.layers[num].trainable = False

x = layers.Flatten()(base_model.output)
output = layers.Dense(1)(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights(r"C:\Users\PMB_MJU\dl_result\2104091638_resnet50\LRP_classifier_best.h5")

# for path in ls_path:
#     print(path)
#     movie = cv2.VideoCapture(str(path))
#     if movie.isOpened() is False:
#         exit()

#     # Define the codec and create VideoWriter object
#     dst_path = os.path.join(dst_dir_path, str(os.path.basename(path))[:-4] + "_predicted.mp4")
#     fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
#     out = cv2.VideoWriter(str(dst_path),fourcc, 6.0, (500,500))
#     total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
#     for num in range(int(total_frame)):
#         flg, img = movie.read()
#         pre_img = np.reshape(img, (1, 500, 500, 3))
#         pre_img = pre_img.astype('float32') / 255
#         stage = model.predict(pre_img)
#         stage = stage[0] * 8
#         dst_img = cv2.putText(img, "stage : " + str(stage), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         out.write(dst_img)

#     out.release()
#     movie.release()

dst_path = r"C:\Users\PMB_MJU\timelapse\resource\dst\210326_col0_msd_4dag\Position006_chan00_croped.avi"
# movie = cv2.VideoCapture(str(ls_path[0]))
movie = cv2.VideoCapture(dst_path)
if movie.isOpened() is False:
    exit()

# Define the codec and create VideoWriter object
total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
out = cv2.VideoWriter(dst_dir_path + ".mp4",fourcc, 6.0, (500,500))

for num in range(int(total_frame)):
    flg, img = movie.read()
    pre_img = np.reshape(img, (1, 500, 500, 3))
    pre_img = pre_img.astype('float32') / 255
    stage = model.predict(pre_img)
    stage = stage[0] * 7
    dst_img = cv2.putText(img, "stage : " + str(stage), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imwrite(dst_dir_path + str(num) + ".png", dst_img)
    out.write(dst_img)
movie.release()
out.release()