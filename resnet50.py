import keras
from keras import applications
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import Image_data_generater_LRP


shape = (500, 500, 3)
batch_size = 6

# keras's data generater
data_gen = Image_data_generater_LRP.ImageDataGenerater('/home/pmb-mju/DL_train_data/train_data_img/LRP_Class_resrc/x40_images_center', 60, img_shape=shape)

callbacks_list = [keras.callbacks.ModelCheckpoint(
                                                filepath='LRP_classifier_best.h5',
                                                monitor='val_loss',
                                                save_best_only=True,
                                                verbose=1),
                    keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=100,
                                                verbose=1),
                    keras.callbacks.ReduceLROnPlateau(
                                                factor=0.1,
                                                monitor='val_loss',
                                                patience=20,
                                                verbose=1),
                    keras.callbacks.CSVLogger('LRP_mid_data.csv', append=False)]

# use application's base model
base_model = applications.ResNet50V2(include_top = False, input_shape = shape)

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
history = model.fit(data_gen.train_generater(batch_size),
                    steps_per_epoch=data_gen.train_num // batch_size,
                    epochs=1000,
                    validation_data=data_gen.val_generate(batch_size),
                    validation_steps=data_gen.val_num//batch_size,
                    callbacks=callbacks_list)

# save last model's information
model.save('LRP_classifier.h5')

# plot learning accuracy log
acc = history.history['mae']
val_acc = history.history['val_mae']
epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()

# save accuracy figure
plt.savefig('LRP_classifier')

# clear figure
plt.clf()

# plot learning loss log
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.legend()

# save loss figure
plt.savefig('LRP_classifier_loss')
