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

ls_path = glob.glob(r'C:\Users\PMB_MJU\timelapse\resource\dst/**/*.avi', recursive=True)

# create model
shape = (500, 500, 3)
base_model = applications.ResNet50(include_top = False, input_shape = shape)

# froze model's layer
# for num, layer in enumerate(base_model.layers):
#     base_model.layers[num].trainable = False

x = layers.Flatten()(base_model.output)
output = layers.Dense(1)(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights(r"C:\Users\PMB_MJU\dl_result\2104132306_resnet50\LRP_classifier_best.h5")

column = list(range(0, 8))
df = pd.DataFrame(columns=column)
df_2 = pd.DataFrame(columns=["path", "stage", "treatment", "time"])
df_3 = pd.DataFrame(columns=["path", "stage", "treatment", "time"]) # timeseries data
df_4 = pd.DataFrame(columns=["path", "stage", "treatment", "time"]) # each stage's data
num_stage = 7
for num_path, i_path in enumerate(ls_path):
    movie = cv2.VideoCapture(str(i_path))
    if movie.isOpened() is False:
        exit()

    treatment = ""
    if "cafe_c22" in i_path:
        treatment = "cafe + c22"
    elif "cafe" in i_path:
        treatment = "cafe"
    elif "kcs1_Cmix" in i_path:
        treatment = "kcs1-5 + Cmix"
    elif "kcs1" in i_path:
        treatment = "kcs1-5"
    elif "kcs2_20" in i_path:
        treatment = "kcs2/20"
    elif "kcs2" in i_path:
        treatment = "kcs2"
    elif "kcs20" in i_path:
        treatment = "kcs20"
    elif "myb93" in i_path:
        treatment = "myb93"
    elif "msd" in i_path:
        treatment = "msd"
    else:
        treatment = "None"

    # Define the codec and create VideoWriter object
    total_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    ls_lrp_time = np.zeros((8))
    time_frame = 0
    time_7 = 0
    Flg_st7 = False
    df_temp = pd.DataFrame(pd.DataFrame(columns=["path", "stage", "treatment", "time"]))
    for num in range(int(total_frame) - 1):
        print("frames " + str(num) +" / " + str(total_frame) + "  " + i_path +  "  " + str(num_path) + "/" + str(len(ls_path)) + "                      ", end="\r")
        flg, img = movie.read()
        pre_img = np.reshape(img, (1, 500, 500, 3))
        pre_img = pre_img.astype('float32') / 255
        stage = model.predict(pre_img)
        if stage[0][0] > 1:
            stage[0][0] = 1
        elif stage[0][0] < 0:
            stage[0][0] = 0
        series = pd.Series([i_path, stage[0][0] * 7, treatment, time_frame], name=num * num_path, index=df_3.columns)
        if stage > 0:
            df_temp = df_temp.append(series)
            time_frame += 30
            stage = math.floor(stage[0] * 7)
            ls_lrp_time[stage] = ls_lrp_time[stage] + 30
        if stage == 7 and Flg_st7 is False:
            Flg_st7 = True
            time_7 = time_frame
            break
    movie.release()
    if Flg_st7 is True:
        # df_temp["time"] = df_temp["time"].to_numpy() / time_7
        # for num_k, time in enumerate(df_temp["time"].tolist()):
        #     time = math.floor(time * 10 ** 2) / (10 ** 2)
        #     df_temp.iloc[num_k, 3] = time

        df_3 = pd.concat([df_3, df_temp])
        for j_stage in range(0, 7):
            stage_series = pd.Series([i_path, j_stage, treatment, len(df_temp[(df_temp["stage"] >= j_stage) & (df_temp["stage"] < (j_stage + 1))]) * 30], name=j_stage * num_path, index=df_4.columns)
            df_4 = df_4.append(stage_series)
    df.loc[str(i_path)] = ls_lrp_time
    for i_stage in range(num_stage): # none stage7
        series = pd.Series([i_path, i_stage, treatment, ls_lrp_time[i_stage]], index=df_2.columns, name=len(df_2.index))
        df_2 = df_2.append(series)
    ls_lrp_time[:] = 0
    Flg_st7 = False
# df.to_csv(str(os.path.join(str(path), "lrp_time.csv")))
df_2.to_csv(str(os.path.join(str(path), "lrp_time2.csv")))
df_3.to_csv(str(os.path.join(str(path), "lrp_time3.csv")))
df_4.to_csv(str(os.path.join(str(path), "lrp_time4.csv")))
