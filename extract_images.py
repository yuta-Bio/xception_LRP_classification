import glob
import os
import shutil

ls_paths = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/200914/dr5_7day_x40_ap6_ex11_int82/**/*.tif", recursive=True)
dst_path = ("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/200914/dr5_7day_x40_ap6_ex11_int82/images")

for num, path in enumerate(ls_paths):
    shutil.move(path, dst_path)
    os.rename(os.path.join(dst_path, os.path.basename(path)), os.path.join(dst_path, os.path.basename(path)[:-4] + '_' + str(num) + '.tif'))