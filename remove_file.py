import os
import glob

ls_path = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/from_timelapse_to_image/**/*.png", recursive=True)

for i in ls_path:
    os.remove(i)