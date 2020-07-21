import glob
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import Image_data_generater_LRP

shape = (512, 512)
eval_num = 50
model = keras.models.load_model('/home/pmb-mju/python_code/xception_LRP_classification/LRP_classifier_best.h5', compile = False)
path = ('/home/pmb-mju/DL_train_data/complete')
# prepare each of stage's paths
stages_path_list = [i for i in glob.glob(path + '/*') if (os.path.isdir(i))]
classed_list = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
class_num_pair  = [(i, j) for i, j in enumerate(classed_list)]  #reference of class's number
num_class = len(class_num_pair)

#class and paths tuples list.  class is 0 to classnum
stages_value = []
for num_stage, i in enumerate(stages_path_list):
    temp_paths = glob.glob(str(i) + '/*.tif' )
    total_value = 0
    for num, j in enumerate(temp_paths):
        img = cv2.imread(
                    str(j), 0)
        crop_rate = 5
        ht = img.shape[0]
        wd = img.shape[1]
        img = img[ht//crop_rate: ht - (ht//crop_rate), wd // crop_rate : wd - (wd // crop_rate)]
        # crop image to square
        if img.shape[0] == img.shape[1]:
            pass
        elif img.shape[0] > img.shape[1]:
            dif = img.shape[0]-img.shape[1]
            img = np.delete(img, np.s_[-(dif//2+1):], 0)
            temp_img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 0)
        else:
            dif = img.shape[1]-img.shape[0]
            img = np.delete(img, np.s_[-(dif//2+1):], 1)
            img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 1)

        img = cv2.resize(img, shape[:2])
        # cv2.namedWindow('temp', cv2.WINDOW_NORMAL)
        # cv2.imshow('temp', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = np.reshape(img, (1, shape[0], shape[1], 1))
        img = img.astype('float16') / 255
        total_value += float(model.predict(img))
        tmep = model.predict(img)
        if num > eval_num or num + 1 == len(temp_paths):
            stages_value.append((total_value / (num + 1)))
            break

print("\n------------------each value of stages------------------------")
print(stages_value)
