import os
import datetime
import shutil
import keras
from keras import applications
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import Image_data_generater_LRP

base_dir = (r'C:\Users\PMB_MJU\dl_result')
basename = datetime.datetime.now().strftime("%y%m%d%H%M") + '_' + str(os.path.basename(str(__file__)))[:-3]
path = os.path.join(base_dir, basename)
if not os.path.isdir(path):
    os.mkdir(path)
shutil.copyfile(__file__, str(os.path.join(path, os.path.basename(__file__))))
shutil.copyfile('Image_data_generater_LRP.py', str(os.path.join(path, 'Image_data_generater_LRP.py')))


shape = (500, 500, 3)
batch_size = 6

# keras's data generater
data_gen = Image_data_generater_LRP.ImageDataGenerater(r'C:\Users\PMB_MJU\x40_images_center_plus', 200, img_shape=shape)

callbacks_list = [keras.callbacks.ModelCheckpoint(
                                                filepath= str(os.path.join(path, 'LRP_classifier_best.h5')),
                                                monitor='val_loss',
                                                save_best_only=True,
                                                verbose=1),
                    keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=20,
                                                verbose=1),
                    keras.callbacks.ReduceLROnPlateau(
                                                factor=0.25,
                                                monitor='val_loss',
                                                patience=10,
                                                verbose=1),
                    keras.callbacks.CSVLogger(str(os.path.join(path, 'LRP_mid_data.csv')))]

# use application's base model
base_model = applications.ResNet50(include_top = False, input_shape = shape)

# froze model's layer
# for num, layer in enumerate(base_model.layers):
#     base_model.layers[num].trainable = False

x = layers.Flatten()(base_model.output)
output = layers.Dense(1)(x)
model = Model(inputs=base_model.input, outputs=output)

# show Deep learning model information
model.summary()

# compile and run
model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='mse', metrics=['mae'])
history = model.fit_generator(data_gen.train_generater(batch_size),
                    steps_per_epoch=data_gen.train_num // batch_size,
                    epochs=1000,
                    validation_data=data_gen.val_generate(batch_size),
                    validation_steps=data_gen.val_num//batch_size,
                    callbacks=callbacks_list)

# save last model's information
model.save(str(os.path.join(path, 'LRP_classifier.h5')))

# plot learning accuracy log
acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'bo', label = 'Training mean_absolute_error')
plt.plot(epochs, val_acc, 'b', label='Validation mean_absolute_error')
plt.legend()

# save accuracy figure
plt.savefig(str(os.path.join(path, 'LRP_classifier_mae.png')))

# clear figure
plt.clf()

# plot learning loss log
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.legend()

# save loss figure
plt.savefig(str(os.path.join(path, 'LRP_classifier_loss.png')))
