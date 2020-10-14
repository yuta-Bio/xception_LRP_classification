import collections
import copy
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


class DL_Process():
    def __init__(self):
        from keras import models
        self.grid_model = models.load_model(
            '/media/suthy/BDiskA/root_img/models/auged_root_segment_model_122419_grid.h5',
            compile=False)
        self.root_model_1024 = models.load_model(
            '/media/suthy/BDiskA/root_img/models/auged_root_segment_model_122619_blur.h5',
            compile=False)
        self.root_model_512 = models.load_model(
            '/media/suthy/BDiskA/root_img/models/auged_root_segment_model_0206_512*512.h5',
            compile=False)

    def grid_recognition(self, img, verbose=False):
        '''return (image's grid interval pixels num) and (1024*1024 size image). this method use DL and so image crop and resize to 1024*1024.
        pix * x = real grid interval(cm).'''

        img = img_to_square(img)
        img = cv2.resize(img, (1024, 1024))
        img = img.astype('float')/255
        img = np.reshape(img, (1, 1024, 1024, 3))
        dst = self.grid_model.predict(img)
        dst = np.reshape(dst, (1024, 1024))
        dst = (np.where(dst > 0.9, 255, 0)).astype('uint8')
        if verbose:
            cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
            cv2.imshow('dst', dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        lines = cv2.HoughLines(dst, 2, np.pi/180, 200)
        if lines is None:
            print('not ditected grid line')
            return False

        # distance is rho value 0 or 90 degree respectively.
        distance = [[], []]
        line_index = [[], []]
        grid_lines = []
        for i, line in enumerate(lines):
            for rho, theta in line:
                if 1.48 <= theta < 1.64:
                    distance[0].append(rho)
                    line_index[0].append(line)
                elif 0 <= theta < 0.09 or 3.05 <= theta < 3.14:
                    distance[1].append(rho)
                    line_index[1].append(line)

        # interval is difference between same angle houghlines.
        interval = [[], []]
        for h, i in enumerate(distance):
            for j in i:
                interval[h].append(i-j)
        interval = list(flatten(interval))
        interval = np.array(interval, dtype=int)
        interval = np.abs(interval)
        interval.ravel()

        # if you want to see interval's graph, remove below coment out
        if verbose:
            plt.hist(interval, bins=100)
            plt.show()

        # convert to hist gram by divided 10.
        classed_interval = interval//10
        classed_interval = classed_interval*10
        co_interval = collections.Counter(classed_interval)

        # Interval_per_grid is distance grid to grid.
        interval_per_grid = 0
        for i in co_interval.most_common():
            if i[0] != 0:
                interval_per_grid = i[0]
                break

        # get more accurately grids interval value.
        interval = interval[(interval <= interval_per_grid+30)
                            & (interval >= interval_per_grid-30)]
        interval_median = np.median(interval)
        if verbose:
            print('grid interval is ' + str(interval_median))
        return interval_median, dst

    def root_recognition(self, img, verbose=False):
        img = img_to_square(img)
        img512_list = divide_img(img, shape=(512, 512))
        img512_list = list(map(lambda x: x.astype('float')/255, img512_list))
        img512_list = list(map(lambda x: np.reshape(x, (1, 512, 512, 3)), img512_list))
        img512_list = list(map(self.root_model_512.predict, img512_list))
        img512_list = list(map(lambda x: np.reshape(x, (512, 512)), img512_list))
        img512_list = list(map(lambda x: (np.where(x > 0.9, 255, 0)).astype('uint8'), img512_list))
        img512      = from_list_to_img(img512_list)
        img1024     = cv2.resize(img, (1024, 1024))
        img1024     = img1024.astype('float')/255
        img1024     = np.reshape(img1024, (1, 1024, 1024, 3))
        img1024     = self.root_model_1024.predict(img1024)
        img1024     = np.reshape(img1024, (1024, 1024))
        img1024     = (np.where(img1024 > 0.9, 255, 0)).astype('uint8')
        img1024     = link_adhered_contours(img1024, img512)
        if verbose:
            imshow(img1024)
        return img1024


def imshow(img, winname='dst', wait_time=0, destroy_flg = False):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    cv2.waitKey(wait_time)
    if destroy_flg:
        cv2.destroyAllWindows()

def img_to_square(img):
    '''this method convert image to square.
    argument: color image
    return  : square image'''
    if img.shape[0] == img.shape[1]:
        pass
    elif img.shape[0] > img.shape[1]:
        dif = img.shape[0]-img.shape[1]
        img = np.delete(img, np.s_[-(dif//2+1):], 0)
        img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 0)
    elif img.shape[0] < img.shape[1]:
        dif = img.shape[1]-img.shape[0]
        img = np.delete(img, np.s_[-(dif//2+1):], 1)
        img = np.delete(img, np.s_[:abs(dif-(dif//2)-1)], 1)
    return img

def divide_img(img, shape=(512, 512), slice_flg=False):
    '''this method divide image as grid. this standard is image converte magnificance.
    use for semantic segmentation
    argment: img is squared image
            shape is destination image shape. it should be power of 2.
    return : divided images list. this sort is from up left to down left.'''
    assert shape[0] & (shape[0] -1) == 0, 'shape is not power of 2'
    img = img_to_square(img)
    dif_list = []
    for i in (512, 1024, 2048):
        dif_list.append([abs(i-img.shape[0]), i])
    dif_list.sort()
    img = cv2.resize(img, (dif_list[0][1], dif_list[0][1]))
    div_num = dif_list[0][1] // shape[0]
    img_list = []
    for i in range(div_num):
        for j in range(div_num):
            if not(i == div_num-1 and slice_flg is True):
                img_list.append(img[i*shape[0]:(i+1)*shape[0], j*shape[0]:(j+1)*shape[0]])
    return img_list

def from_list_to_img(img_list):
    '''this method replicate image from divided image list.
    image list should be sorted from up left to down left.
    argument: img_list is image list
                shape is estimated replicated image shape
    return:   dst_img is replicated image.'''
    len_list = len(img_list)
    ori_size = (img_list[0].shape)[0] #512
    img_num = int(math.sqrt(len_list)) #row or column images number
    if len(img_list[0].shape) == 2:
        dst_img = np.zeros((ori_size*img_num, ori_size*img_num), 'uint8')
    else:
        dst_img = np.zeros((ori_size*img_num, ori_size*img_num, img_list[0].shape[3]), 'uint8')
    for i in range(img_num):
        for j in range(img_num):
            dst_img[i*ori_size:(i+1)*ori_size, j*ori_size:(j+1)*ori_size] = img_list[(i*img_num)+j]
    return dst_img

def flatten(f_list):
    for el in f_list:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def marged_contour_to_bbox(contours):
    bboxes = []
    for i in contours:
        x, y, wd, hg = cv2.boundingRect(i)
        y_var = [y-25, y+hg+25]
        x_var = [x-10, x+wd+10]
        if not bboxes:
            bboxes.append([y_var, x_var])
        else:
            del_nums = []
            for num, j in enumerate(bboxes):
                # neighbored bounding box is marged
                if ((j[0][0] < y_var[0] < j[0][1]) or (j[0][0] < y_var[1] < j[0][1]) or ((y_var[0] < j[0][0]) and (y_var[1] > j[0][1])))\
                    and ((j[1][0] < x_var[0] < j[1][1]) or (j[1][0] < x_var[1] < j[1][1]) or ((x_var[0] < j[1][0]) and (x_var[1] > j[1][1]))):
                    if y_var[0] > j[0][0]:
                        y_var[0] = j[0][0]
                    if y_var[1] < j[0][1]:
                        y_var[1] = j[0][1]
                    if x_var[0] > j[1][0]:
                        x_var[0] = j[1][0]
                    if x_var[1] < j[1][1]:
                        x_var[1] = j[1][1]
                    #marged bounding box number is append and pop out.
                    del_nums.append(num)
            if del_nums:
                for j in reversed(del_nums):
                    bboxes.pop(j)
            bboxes.append([y_var, x_var])
    return bboxes

def link_adhered_contours(ori_img, detail_img):
    '''this method build image from ori_img and detail_img.
    Detail image is included detailed discription, and extra garbage line.
    So this method is built from original accurate line but rough image. And attach detailed line from detail_img to ori_img.'''
    detail_img = cv2.resize(detail_img, ori_img.shape[:2])
    contours, _ = cv2.findContours(detail_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    black_img = np.zeros(ori_img.shape, 'uint8')
    for contour_i in contours:
        temp_img = copy.copy(black_img)
        temp_img = cv2.drawContours(temp_img, [contour_i], 0, 255, -1)
        temp_img = cv2.dilate(temp_img, np.ones((15, 15), 'uint8'))
        if cv2.countNonZero(cv2.bitwise_and(ori_img, temp_img)) > 0:
            ori_img = cv2.drawContours(ori_img, [contour_i], 0, 255, -1)
    return ori_img

def measure_len(img):
    '''this method measure line's length from skeletonized image.
    # Arguments
    - img[two dimention ndarray] : bicolor skeletonized image.
    # Returns
    - length[int] : length as measured by pixel.
    '''
    length = 0
    line_img = np.zeros(img.shape[:2], dtype='uint8')
    kernel1_1   = np.array([[0, 0, 0],
                            [1, 1, 0],
                            [0, 0, 0]], dtype='uint8')
    kernel1_2   = np.rot90(kernel1_1)
    kernel1_3   = np.rot90(kernel1_2)
    kernel1_4   = np.rot90(kernel1_3)
    kernel_list1 = [kernel1_1, kernel1_2, kernel1_3, kernel1_4]
    for kernel in kernel_list1:
        line_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS,
                                    kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        length += cv2.countNonZero(line_img) / 2

    kernel2_1   = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype='uint8')
    kernel2_2   = np.rot90(kernel2_1)
    kernel2_3   = np.rot90(kernel2_2)
    kernel2_4   = np.rot90(kernel2_3)
    kernel_list2 = [kernel2_1, kernel2_2, kernel2_3, kernel2_4]
    for kernel in kernel_list2:
        line_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS,
                                    kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        length += cv2.countNonZero(line_img) * (math.sqrt(2) / 2)

    # kernel2_1   = np.array([[-1, -1, 1],
    #                         [-1, 1, -1],
    #                         [1, -1, -1]], dtype='uint8')
    # kernel2_2   = np.rot90(kernel2_1)
    # kernel_list2 = [kernel2_1, kernel2_2]
    # line_img = np.zeros(img.shape[:2], dtype='uint8')
    # for kernel in kernel_list2:
    #     line_img = cv2.bitwise_or(cv2.morphologyEx(img, cv2.MORPH_HITMISS,
    #                                 kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), line_img)
    # length += cv2.countNonZero(line_img) * math.sqrt(2)

    # kernel3_1   = np.array([[-1, -1, -1],
    #                         [1, 1, -1],
    #                         [-1, -1, 1]], dtype='uint8')
    # kernel3_2   = np.array([[-1, -1, 1],
    #                         [1, 1, -1],
    #                         [-1, -1, -1]], dtype='uint8')
    # kernel3_3   = np.rot90(kernel3_1)
    # kernel3_4   = np.rot90(kernel3_2)
    # kernel3_5   = np.rot90(kernel3_3)
    # kernel3_6   = np.rot90(kernel3_4)
    # kernel3_7   = np.rot90(kernel3_5)
    # kernel3_8   = np.rot90(kernel3_6)
    # kernel_list3 = [kernel3_1, kernel3_2, kernel3_3, kernel3_4, kernel3_5, kernel3_6, kernel3_7, kernel3_8]
    # line_img = np.zeros(img.shape[:2], dtype='uint8')
    # for kernel in kernel_list3:
    #     line_img = cv2.bitwise_or(cv2.morphologyEx(img, cv2.MORPH_HITMISS,
    #                                 kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), line_img)
    # length += cv2.countNonZero(line_img) * ((math.sqrt(2)/2)+(1/2))

    # kernel4_1   = np.array([[-1, -1, -1],
    #                         [1, 1, -1],
    #                         [-1, -1, -1]], dtype='uint8')
    # kernel4_2   = np.rot90(kernel4_1)
    # kernel4_3   = np.rot90(kernel4_2)
    # kernel4_4   = np.rot90(kernel4_3)
    # kernel_list4 = [kernel4_1, kernel4_2, kernel4_3, kernel4_4]
    # line_img = np.zeros(img.shape[:2], dtype='uint8')
    # for kernel in kernel_list4:
    #     line_img = cv2.bitwise_or(cv2.morphologyEx(img, cv2.MORPH_HITMISS,
    #                                 kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), line_img)
    # length += cv2.countNonZero(line_img) * (1/2)

    # kernel5_1   = np.array([[-1, -1, -1],
    #                         [-1, 1, -1],
    #                         [1, -1, -1]], dtype='uint8')
    # kernel5_2   = np.rot90(kernel5_1)
    # kernel5_3   = np.rot90(kernel5_2)
    # kernel5_4   = np.rot90(kernel5_3)
    # kernel_list5 = [kernel5_1, kernel5_2, kernel5_3, kernel5_4]
    # line_img = np.zeros(img.shape[:2], dtype='uint8')
    # for kernel in kernel_list5:
    #     line_img = cv2.bitwise_or(cv2.morphologyEx(img, cv2.MORPH_HITMISS,
    #                                 kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), line_img)
    # length += cv2.countNonZero(line_img) * (math.sqrt(2)/2)

    # kernel6_1   = np.array([[-1, 1, -1],
    #                         [1, 1, 1],
    #                         [-1, -1, -1]], dtype='uint8')
    # kernel6_2   = np.rot90(kernel6_1)
    # kernel6_3   = np.rot90(kernel6_2)
    # kernel6_4   = np.rot90(kernel6_3)
    # kernel_list6 = [kernel6_1, kernel6_2, kernel6_3, kernel6_4]
    # line_img = np.zeros(img.shape[:2], dtype='uint8')
    # for kernel in kernel_list6:
    #     line_img = cv2.bitwise_or(cv2.morphologyEx(img, cv2.MORPH_HITMISS,
    #                                 kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), line_img)
    # length += cv2.countNonZero(line_img) * (3/2)

    # kernel7_1   = np.array([[-1, -1, 1],
    #                         [1, 1, 1],
    #                         [-1, -1, -1]], dtype='uint8')
    # kernel7_2   = np.array([[-1, -1, -1],
    #                         [1, 1, 1],
    #                         [-1, -1, 1]], dtype='uint8')
    # kernel7_3   = np.rot90(kernel7_1)
    # kernel7_4   = np.rot90(kernel7_2)
    # kernel7_5   = np.rot90(kernel7_3)
    # kernel7_6   = np.rot90(kernel7_4)
    # kernel7_7   = np.rot90(kernel7_5)
    # kernel7_8   = np.rot90(kernel7_6)
    # kernel_list7 = [kernel7_1, kernel7_2, kernel7_3, kernel7_4]
    # line_img = np.zeros(img.shape[:2], dtype='uint8')
    # for kernel in kernel_list7:
    #     line_img = cv2.bitwise_or(cv2.morphologyEx(img, cv2.MORPH_HITMISS,
    #                                 kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0), line_img)
    # length += cv2.countNonZero(line_img) * (math.sqrt(2)/2 + 1)
    return length

def image_substraction(img1, img2):
    '''this method define 0 or 255 image's substraction.
    both image shape should be same and 0 - 255 is 0, not -255.
    use for bicolor image.'''
    dif_img = img1.astype('int') - img2.astype('int')
    dif_img = np.where(dif_img > 0, dif_img, 0)
    dif_img = dif_img.astype('uint8')
    return dif_img

if __name__ == "__main__":
    src_img = cv2.imread('/home/suthy/python/ktest/line.png', 0)
    # divide_img_list = divide_img(src_img)
    # src_img = from_list_to_img(divide_img_list)
    m_length = measure_len(src_img)
    print(m_length)
    # imshow(src_img)
