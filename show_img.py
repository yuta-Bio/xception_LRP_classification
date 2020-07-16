import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


train_dir = ""

fnames = [os.path.join(train_dir, fname)
                    for fname in os.listdir(train_dir)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(512,512))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=360,
                    width_shift_range=0.5,
                    height_shift_range=0.5,
                    shear_range=0.5,
                    zoom_range=0.5,
                    horizontal_flip=True,
                    vertical_flip=True)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
