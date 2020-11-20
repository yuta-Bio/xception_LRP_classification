import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = ('/home/pmb-mju/python_code/xception_LRP_classification/LRP_mid_data.csv')
matrix = pd.read_csv(path)
epochs = matrix['epoch'].to_numpy()

mae = matrix['mae'].to_numpy()
val_mae = matrix['val_mae'].to_numpy()
plt.plot(epochs, mae, 'bo', label = 'Training mae')
plt.plot(epochs, val_mae, 'b', label='Validation mae')
plt.legend()

# save accuracy figure
plt.savefig('LRP_classifier_mae.png')

# clear figure
plt.clf()

loss = matrix['loss'].to_numpy()
val_loss = matrix['val_loss'].to_numpy()
plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.legend()
plt.savefig('LRP_classifier_loss.png')
