import os
import glob

ls_path = glob.glob("/home/pmb-mju/DL_train_data/train_data_img/calus/renamed/src/*.TIF")

for num, path in enumerate(ls_path):
    # if (num % 6) == 0:
    #     pass
    # else:
    #     os.remove(path)
    os.remove(path)
