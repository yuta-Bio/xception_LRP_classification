import os
import datetime
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# create log folder
base_dir = ('/home/pmb-mju/dl_result')
basename = datetime.datetime.now().strftime("%y%m%d%H%M")
path = os.path.join(base_dir, basename)
if not os.path.isdir(path):
    os.mkdir(path)
shutil.copyfile(__file__, str(os.path.join(path, os.path.basename(__file__))))

# folders
path1 = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_1st")
path2 = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_2nd")
path3 = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_3rd")
path4 = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_4th")

# list paths
ls_paths = [path1, path2, path3, path4]

# glob path1's images
ls_path_image = glob.glob(str(path1) + "/**/*.tif")

# extract filename from ls_path_image
col_names = [str(os.path.basename(path)) for path in ls_path_image]

# list stages (subfolder)
ls_stages = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]
row_names = ["1st", "2nd", "3rd", "4th"]

# create dataframe (initial stage is 8)
zeros_array = np.zeros((len(col_names), len(row_names)))
zeros_array[:, :] = 8
df = pd.DataFrame(zeros_array, col_names, row_names)

# change dataframe's value
for i in range(4):
    for num_j, name_image_file in enumerate(df.index):
        for num_stage, k in enumerate(ls_stages):
            if os.path.isfile(str(os.path.join(ls_paths[i], k, str(name_image_file)))):
                df.iloc[num_j, i] = num_stage
                break
            else:
                pass

# save file to log folder
df.to_csv(str(os.path.join(path, 'aggregate_human_classification.csv')))
