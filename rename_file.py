import os
import glob

ls_path = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center_plus/**/*.tif", recursive=True)

for num, i in enumerate(ls_path):
    src_base = str(os.path.basename(str(i)))
    dst_base = str(num) + src_base
    dst_path = os.path.join(os.path.dirname(i), dst_base)
    os.rename(str(i), str(dst_path))
