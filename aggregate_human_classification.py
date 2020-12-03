import os
import datetime
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# base_dir = ('/home/pmb-mju/dl_result')
# basename = datetime.datetime.now().strftime("%y%m%d%H%M")
# path = os.path.join(base_dir, basename)
# if not os.path.isdir(path):
#     os.mkdir(path)
# shutil.copyfile(__file__, str(os.path.join(path, os.path.basename(__file__))))

path1 = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_1st")
path2 = ("")
path3 = ("")
path4 = ("")

ls_path_image = glob.glob(str(path1) + "/**/*.tif")
col_names = [str(os.path.basename(path)) for path in ls_path_image]
ls_stages = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]
row_names = ["1st", "2nd", "3rd", "4th"]
zeros_array = np.zeros((len(col_names), len(row_names)))
zeros_array[:, :] = 8
df = pd.DataFrame(zeros_array, col_names, row_names)

for i in range(1, 5):
    for name_image_file in df.index:
        stage = 8
        for k in range(len(ls_stages)):
            if os.path.isfile(str(os.path.join)):
            pass
print(df)