import os
import shutil
import random
import glob

ls_path = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_5th/stage7/*.tif")
path_dst = "/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus_5th/images"
random_list = random.sample(ls_path, 50)
for path in random_list:
    shutil.move(path, path_dst)