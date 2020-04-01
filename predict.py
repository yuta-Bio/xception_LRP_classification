import keras
from keras import applications
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2

shape = (512, 512, 3)
batch_size = 2
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=360,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    "/media/suthy/BDiskA/lateral_root_primordium_image/train",
                    target_size=shape[:2],
                    batch_size=batch_size,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    "/media/suthy/BDiskA/lateral_root_primordium_image/test",
                    target_size=shape[:2],
                    batch_size=batch_size,
                    class_mode='categorical')

callbacks_list = [keras.callbacks.ModelCheckpoint(
                                                filepath='LRP_classifier_best.h5',
                                                monitor='val_loss',
                                                save_best_only=True,
                                                verbose=1),
                    keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=400,
                                                mode='min',
                                                verbose=1),
                    keras.callbacks.ReduceLROnPlateau(
                                                monitor='val_loss',
                                                factor=0.1,
                                                patience=300,
                                                verbose=1)]

base_model = applications.xception.Xception(weights=None, include_top=False, input_shape=shape)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(1000, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator,
                    steps_per_epoch=70//batch_size,
                    epochs=1000,
                    validation_data=test_generator,
                    validation_steps=12//batch_size,
                    callbacks=callbacks_list)
model.save('LRP_classifier.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'bo', label = 'Training loss')
plt.plot(epochs, val_acc, 'b', label='Validation_loss')
plt.legend()
plt.savefig('LRP_classifier')
