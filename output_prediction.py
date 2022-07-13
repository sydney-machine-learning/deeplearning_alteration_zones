import numpy as np
import random
from pyrsgis import raster
from pyrsgis.ml import array_to_chips
from math import floor
from tensorflow import keras
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

Model_location = r"D:\Internship\UNSW\deeplearning-alterationzones\models\models\LANDSAT8_DATA_CNN_ADAM_MODEL"
NAME = "LANDSAT8_DATA_ADAM_MODEL"

model = keras.models.load_model(Model_location)

# Loading and normalizing a new multispectral image
dsPre, featuresPre = raster.read('Datasets/BrokenHill_Landsat8.tif')
# featuresPre = featuresPre.astype(float)
featuresPre = np.float16(featuresPre)

for i in range(featuresPre.shape[0]):
    bandMinPre = featuresPre[i][:][:].min()
    bandMaxPre = featuresPre[i][:][:].max()
    bandRangePre = bandMaxPre-bandMinPre
    for j in range(featuresPre.shape[1]):
        for k in range(featuresPre.shape[2]):
            featuresPre[i][j][k] = (featuresPre[i][j][k]-bandMinPre)/bandRangePre

# Generating image chips from the array
new_features = array_to_chips(featuresPre, x_size=7, y_size=7)


##### Some Preprocessing (Only for MLP Code) ######
#new_features = np.reshape(new_features, (new_features.shape[0], new_features.shape[1]*new_features.shape[2]*new_features.shape[3]))

newPredicted = model.predict(new_features)

prediction = np.reshape(newPredicted.argmax(axis=1), (dsPre.RasterYSize, dsPre.RasterXSize))

outFile = 'Mapped Files/BrokenHill_{}.tif'.format(NAME)

raster.export(prediction, dsPre, filename=outFile, dtype='float')