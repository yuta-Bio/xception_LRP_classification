import copy
import glob
import random
import os
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance

class ImageDataGenerater(object):
    def __init__(self, src_path, val_num, img_shape=(1024, 1024, 3)):
        self.img_shape = img_shape

        self.crop_rate = 3.5
        # prepare each of stage's paths
        stages_path_list = [i for i in glob.glob(src_path + '/*') if (os.path.isdir(i))]
        classed_list = [i for i in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, i))]
        self.class_num_pair  = []  #reference of class's number

        #class and paths tuples list.  class is 0 to classnum
        self.class_path_pairs = []
        stages_path_list.sort(key = lambda x: int(x[-1]))
        for num_stage, i in enumerate(stages_path_list):
            temp_path = glob.glob(str(i) + '/*' )
            self.class_num_pair.append((num_stage, i))
            for j in temp_path:
                self.class_path_pairs.append((num_stage, j))

        self.num_class = len(self.class_num_pair) - 1

        #shuffle class and path's list for randomize
        random.seed(1)
        random.shuffle(self.class_path_pairs)

        # ensure memory area
        self.src_img     = np.zeros(self.img_shape, 'uint8')
        self.temp_img    = np.zeros(self.img_shape, 'uint8')
        self.pre_src_img = np.zeros(self.img_shape, 'uint8')

        # define training and validation number
        self.train_num  = len(self.class_path_pairs) - val_num
        self.val_num = val_num

        # images and numbers list
        self.train_img_list = []
        self.train_class_list = []
        self.val_img_list   = []
        self.val_class_list   = []

        # separete data to training and validation
        flg_imread = 0
        if img_shape[2] == 1:
            flg_imread = 0
        else: flg_imread = 1

        for num, i in enumerate(self.class_path_pairs):
            print('total num ', str(len(self.class_path_pairs)), 'current num ', str(num + 1), i[1] + "                       ", end = '\r')
            temp_src_img = cv2.imread(
                                str(i[1]), flg_imread)

            # append data to training list
            if num >= val_num:
                self.train_img_list.append(temp_src_img)
                self.train_class_list.append(i[0])

            # append data to validation list
            else:

                # crop image
                ht = temp_src_img.shape[0]
                wd = temp_src_img.shape[1]
                temp_src_img = temp_src_img[int(ht/self.crop_rate): int(ht - (ht/self.crop_rate)), int(wd / self.crop_rate) : int(wd - (wd / self.crop_rate))]
                #reshape image to square
                if temp_src_img.shape[0] == temp_src_img.shape[1]:
                    pass
                elif temp_src_img.shape[0] > temp_src_img.shape[1]:
                    dif = temp_src_img.shape[0]-temp_src_img.shape[1]
                    temp_src_img = np.delete(temp_src_img, np.s_[-(dif//2+1):], 0)
                    temp_img = np.delete(temp_src_img, np.s_[:abs(dif-(dif//2)-1)], 0)
                else:
                    dif = temp_src_img.shape[1]-temp_src_img.shape[0]
                    temp_src_img = np.delete(temp_src_img, np.s_[-(dif//2+1):], 1)
                    temp_src_img = np.delete(temp_src_img, np.s_[:abs(dif-(dif//2)-1)], 1)

                # append list
                self.val_img_list.append(cv2.resize(temp_src_img, img_shape[:2]))
                self.val_class_list.append(i[0])

    def train_generater(self, batch_size):
        inputs = np.zeros((batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]), 'float32')
        # targets = np.zeros((batch_size, self.num_class), 'float32')
        targets = np.zeros((batch_size), 'float32')

        while True:
            batch_count = 0
            for self.pre_src_img, i_class_num in zip(self.train_img_list, self.train_class_list):
                if batch_count == batch_size:
                    batch_count = 0
                    yield inputs, targets

                # crop image to square
                if self.pre_src_img.shape[0] == self.pre_src_img.shape[1]:
                    pass
                elif self.pre_src_img.shape[0] > self.pre_src_img.shape[1]:
                    dif = self.pre_src_img.shape[0]-self.pre_src_img.shape[1]
                    self.pre_src_img = np.delete(self.pre_src_img, np.s_[-(dif//2+1):], 0)
                    temp_img = np.delete(self.pre_src_img, np.s_[:abs(dif-(dif//2)-1)], 0)
                else:
                    dif = self.pre_src_img.shape[1]-self.pre_src_img.shape[0]
                    self.pre_src_img = np.delete(self.pre_src_img, np.s_[-(dif//2+1):], 1)
                    self.pre_src_img = np.delete(self.pre_src_img, np.s_[:abs(dif-(dif//2)-1)], 1)

                # random fliping
                flip_case = random.randint(0, 3)
                if flip_case == 0:
                    pass
                elif flip_case == 1:
                    self.pre_src_img = cv2.flip(self.pre_src_img, 0)

                elif flip_case == 2:
                    self.pre_src_img = cv2.flip(self.pre_src_img, 1)

                elif flip_case == 3:
                    self.pre_src_img = cv2.flip(self.pre_src_img, -1)

                # random rotation
                theta = random.randint(0, 360)
                oy, ox = self.pre_src_img.shape[0]/2, self.pre_src_img.shape[1]/2
                R = cv2.getRotationMatrix2D((ox, oy), theta, 1.0)
                self.pre_src_img = cv2.warpAffine(self.pre_src_img, R, (int(ox*2), int(oy*2)), flags=cv2.INTER_NEAREST)

                # random translation
                move_x = random.randint(-1 * self.pre_src_img.shape[1] // 9, self.pre_src_img.shape[1] // 9)
                move_y = random.randint(-1 * self.pre_src_img.shape[0] // 9, self.pre_src_img.shape[0] // 9)
                self.pre_src_img = cv2.warpAffine(self.pre_src_img, np.float32([[1, 0, move_x], [0, 1, move_y]]), (self.pre_src_img.shape[1], self.pre_src_img.shape[0]))

                # crop image
                ht = self.pre_src_img.shape[0]
                wd = self.pre_src_img.shape[1]
                self.pre_src_img = self.pre_src_img[int(ht/self.crop_rate): int(ht - (ht/self.crop_rate)), int(wd / self.crop_rate) : int(wd - (wd / self.crop_rate))]

                # random magnificance
                range_img = random.uniform(0.8, 1.2)
                self.pre_src_img    = cv2.resize(self.pre_src_img, (int(self.pre_src_img.shape[0] * range_img), int(self.pre_src_img.shape[0] * range_img)), interpolation=cv2.INTER_NEAREST)

                self.src_img = cv2.resize(self.pre_src_img, self.img_shape[:2])

                # image process at whole image
                alpha = random.uniform(0.7, 1.3)
                pil_temp = Image.fromarray(self.src_img)
                con_temp = ImageEnhance.Contrast(pil_temp)
                pil_temp = con_temp.enhance(alpha)
                
                bri_temp = ImageEnhance.Brightness(pil_temp)
                pil_temp = bri_temp.enhance(random.uniform(0.8, 1.2))

                con_temp1 = ImageEnhance.Sharpness(pil_temp)
                pil_temp = con_temp1.enhance(random.uniform(0.8, 1.2))
                # pil_temp = pil_temp.convert("L")

                self.src_img = np.array(pil_temp)
                self.src_img = np.clip(self.src_img, 0, 255).astype(np.uint8)

                # random resolution
                kernel_num = random.randint(1, 4) * 2 - 1
                self.src_img = cv2.blur(self.src_img, (kernel_num, kernel_num))

                # draw rectangle
                rec_freq = random.randint(0, 5)
                for k in range(rec_freq):

                    # add black rectangle
                    x = random.randint(0, self.img_shape[0]-1)
                    y = random.randint(0, self.img_shape[0]-1)
                    if x+(self.img_shape[0]//3*2) <= self.img_shape[0]:
                        h = random.randint(x+1, x+self.img_shape[0]//3*2)
                    else:
                        h = random.randint(x+1, self.img_shape[0])
                    if y+(self.img_shape[0]//3*2) <= self.img_shape[0]:
                        w = random.randint(y+1, y+self.img_shape[0]//3*2)
                    else:
                        w = random.randint(y+1, self.img_shape[0])

                    rec_img = copy.copy(self.temp_img)
                    rand_color = random.randint(0, 150)
                    rec_img = cv2.rectangle(rec_img, (x, y), (h, w), (rand_color, rand_color, rand_color), -1)
                    self.src_img = cv2.subtract(self.src_img, rec_img)

                # cutout image
                hg_fw = random.randint(0, 100)
                hg_rv = random.randint(0, 100)
                wd_fw = random.randint(0, 100)
                wd_rv = random.randint(0, 100)
                self.src_img[0:hg_fw, :] = 0
                self.src_img[self.src_img.shape[0] - hg_rv:self.src_img.shape[0], :] = 0
                self.src_img[0:wd_fw, :] = 0
                self.src_img[:, self.src_img.shape[1] - wd_rv:self.src_img.shape[1]] = 0

                # reshape data to input shape
                inputs[batch_count] = (self.src_img.astype('float32') / 255.).reshape(self.img_shape)


                # targets[batch_count] = 0
                # targets[batch_count, i_class_num] = 1
                targets[batch_count] = i_class_num / self.num_class
                batch_count += 1

    def val_generate(self, batch_size):
        inputs = np.zeros((batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]), 'float32')
        # targets = np.zeros((batch_size, 1), 'float32')
        # targets = np.zeros((batch_size, self.num_class), 'float32')
        targets = np.zeros((batch_size), 'float32')
        while True:
            batch_count = 0
            for self.src_img, i_class_num in zip(self.val_img_list, self.val_class_list):
                if batch_count == batch_size:
                    batch_count = 0
                    yield inputs, targets
                inputs[batch_count] = (self.src_img.astype('float32') / 255.).reshape(self.img_shape)

                # targets[batch_count] = 0
                # targets[batch_count, i_class_num] = 1
                targets[batch_count] = i_class_num / self.num_class
                batch_count += 1



if __name__ == "__main__":
    path = (r'C:\Users\PMB_MJU\x40_images_center_plus')
    shape = (500, 500, 3)
    data_gen = ImageDataGenerater(path, 30, img_shape=shape)

    print(data_gen.class_num_pair)
    print('train num', str(len(data_gen.train_class_list)))
    print('val num', str(len(data_gen.val_class_list)))
    for num, i in enumerate(data_gen.train_generater(1)):
        print(i[1])
        img = (i[0] * 255).astype('uint8').reshape(shape)
        cv2.namedWindow('temp', cv2.WINDOW_NORMAL)
        cv2.imshow('temp', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(np.mean(i[0]), np.max(i[0]), np.min(i[0]))

