import os
import glob
import shutil
import datetime
import math
import pandas as pd
import keras
from keras import layers, Model, applications
import tensorflow as tf
import numpy as np
import cv2

base_dir = (R"C:\Users\PMB_MJU\dl_result")
basename = datetime.datetime.now().strftime("%y%m%d%H%M") + '_' + str(os.path.basename(str(__file__)))[:-3]
path = os.path.join(base_dir, basename)
if not os.path.isdir(path):
    os.mkdir(path)
shutil.copyfile(__file__, str(os.path.join(path, os.path.basename(__file__))))

ls_path = glob.glob(r'C:\Users\PMB_MJU\timelapse\resource\dst/**/*.mp4', recursive=True)

# create model
shape = (500, 500, 3)
base_model = applications.ResNet50(include_top = False, input_shape = shape)

# froze model's layer
# for num, layer in enumerate(base_model.layers):
#     base_model.layers[num].trainable = False

x = layers.Flatten()(base_model.output)
output = layers.Dense(1)(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights(r"C:\Users\PMB_MJU\dl_result\2103181525_resnet50\LRP_classifier_best.h5")

column = list(range(0, 8))
df = pd.DataFrame(columns=column)
df_2 = pd.DataFrame(columns=["path", "stage", "treatment", "time"])
df_3 = pd.DataFrame(columns=["path", "stage", "treatment", "time"])
num_stage = 7
for i_path in ls_path:
    print(i_path)
    movie = cv2.VideoCapture(str(i_path))
    if movie.isOpened() is False:
        exit()

    col_treatment = ""
    if "msd" in i_path:
        col_treatment = "msd"
    elif "cafe_c22" in i_path:
        col_treatment = "cafe+c22"
    elif "cafe" in i_path:
        col_treatment = "cafe"
    elif "kcs1" in i_path:
        col_treatment = "kcs1-5"
    elif "kcs2_20" in i_path:
        col_treatment = "kcs2/20"
    elif "kcs2" in i_path:
        col_treatment = "kcs2"
    elif "kcs20" in i_path:
        col_treatment = "kcs20"
    else:
        col_treatment = "None"

    # Define the codec and create VideoWriter object
    total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    ls_lrp_time = np.zeros((8))
    time_frame = 0
    for num in range(int(total_frame)):
        print("frames " + str(num) +" / " + str(total_frame) , end="\r")
        flg, img = movie.read()
        pre_img = np.reshape(img, (1, 500, 500, 3))
        pre_img = pre_img.astype('float32') / 255
        stage = model.predict(pre_img)
        series = pd.Series([i_path, stage[0][0] * 7, col_treatment, time_frame], name=len(df_3.index), index=df_3.columns)
        df_3 = df_3.append(series)
        stage = math.floor(stage[0] * 7)
        if stage > 7:
            stage = 7
        elif stage < 0:
            stage = 0
        ls_lrp_time[stage] = ls_lrp_time[stage] + 30
        time_frame += 30
    movie.release()
    df.loc[str(i_path)] = ls_lrp_time
    for i_stage in range(num_stage): # none stage7
        series = pd.Series([i_path, i_stage, col_treatment, ls_lrp_time[i_stage]], index=df_2.columns, name=len(df_2.index))
        df_2 = df_2.append(series)
    ls_lrp_time[:] = 0
# df.to_csv(str(os.path.join(str(path), "lrp_time.csv")))
df_2.to_csv(str(os.path.join(str(path), "lrp_time2.csv")))
df_3.to_csv(str(os.path.join(str(path), "lrp_time3.csv")))
