import keras
from keras import applications
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import Image_data_generater_LRP


shape = (512, 512, 1)
batch_size = 2
# train_datagen = ImageDataGenerator(
#                     rescale=1./255,
#                     rotation_range=360,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     shear_range=0.2,
#                     zoom_range=0.2,
#                     horizontal_flip=True,
#                     vertical_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#                     "/home/pmb-mju/DL_train_data/LRP_Class_resrc/train",
#                     target_size=shape[:2],
#                     batch_size=batch_size,
#                     class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
#                     "/home/pmb-mju/DL_train_data/LRP_Class_resrc/test",
#                     target_size=shape[:2],
#                     batch_size=batch_size,
#                     class_mode='categorical')

data_gen = Image_data_generater_LRP.ImageDataGenerater('/media/suthy/BDiskA/LRP_Class_resrc/lateral_root_primordium_image_ori/complete2', 100, img_shape=shape)

callbacks_list = [keras.callbacks.ModelCheckpoint(
                                                filepath='LRP_classifier_best.h5',
                                                monitor='val_loss',
                                                save_best_only=True,
                                                verbose=1),
                    keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=30,
                                                mode='min',
                                                verbose=1),
                    keras.callbacks.ReduceLROnPlateau(
                                                monitor='val_loss',
                                                factor=0.1,
                                                patience=10,
                                                verbose=1),
                    keras.callbacks.CSVLogger('LRP_mid_data.csv', append=False)]

base_model = applications.xception.Xception(include_top=False, input_shape=shape, weights = None)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(1000, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
output = layers.Dense(data_gen.num_class, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for num, layer in enumerate(model.layers):
    model.layers[num].trainable = True

model.summary()
model.compile(Adam(), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit_generator(data_gen.train_generater(batch_size),
                    steps_per_epoch=data_gen.train_num // batch_size,
                    epochs=200,
                    validation_data=data_gen.val_generate(batch_size),
                    validation_steps=data_gen.val_num//batch_size,
                    callbacks=callbacks_list)
model.save('LRP_classifier.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.savefig('LRP_classifier')
