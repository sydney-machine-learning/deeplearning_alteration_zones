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

# models = []

# file_locations = [r"D:\Internship\UNSW\Alteration_Zones\models\1_conv_1_kernel", r"D:\Internship\UNSW\Alteration_Zones\models\1_conv_2_kernel",
# r"D:\Internship\UNSW\Alteration_Zones\models\1_conv_3_kernel", r"D:\Internship\UNSW\Alteration_Zones\models\2_conv_1_kernel",
# r"D:\Internship\UNSW\Alteration_Zones\models\2_conv_2_kernel", r"D:\Internship\UNSW\Alteration_Zones\models\2_conv_3_kernel",
# r"D:\Internship\UNSW\Alteration_Zones\models\Original_model"]

# NAME = ["1_conv_1_kernel", "1_conv_2_kernel", "1_conv_3_kernel", "2_conv_1_kernel", "2_conv_2_kernel", "2_conv_3_kernel", "Original_model"]

# for i in range(7):
#     model = keras.models.load_model(file_locations[i])
#     models.append(model)
# model = keras.models.load_model(r"D:\Internship\UNSW\Alteration_Zones\models_and_results\ASTER_Original_Model\ASTER_Original_Model")

# # Loading and normalizing a new multispectral image
# dsPre, featuresPre = raster.read('BrokenHill_ASTER.tif')
# # featuresPre = featuresPre.astype(float)
# featuresPre = np.float16(featuresPre)

# for i in range(featuresPre.shape[0]):
#     bandMinPre = featuresPre[i][:][:].min()
#     bandMaxPre = featuresPre[i][:][:].max()
#     bandRangePre = bandMaxPre-bandMinPre
#     for j in range(featuresPre.shape[1]):
#         for k in range(featuresPre.shape[2]):
#             featuresPre[i][j][k] = (featuresPre[i][j][k]-bandMinPre)/bandRangePre

# # Generating image chips from the array
# new_features = array_to_chips(featuresPre, x_size=7, y_size=7)

# #for i in range(len(models)):
#     # Predicting new data and exporting the probability raster
# newPredicted = model.predict(new_features)

# prediction = np.reshape(newPredicted.argmax(axis=1), (dsPre.RasterYSize, dsPre.RasterXSize))

# outFile = 'BrokenHill_ASTER_Original_AlterationMap_CNN.tif'.format()

# raster.export(prediction, dsPre, filename=outFile, dtype='float')















# 
# model = keras.models.load_model(r"D:\Internship\UNSW\Alteration_Zones\models_and_results\ASTER_Oversampled_Model\ASTER_Oversampled_Model")

# # Loading and normalizing a new multispectral image
# dsPre, featuresPre = raster.read('BrokenHill_ASTER.tif')
# # featuresPre = featuresPre.astype(float)
# featuresPre = np.float16(featuresPre)

# for i in range(featuresPre.shape[0]):
#     bandMinPre = featuresPre[i][:][:].min()
#     bandMaxPre = featuresPre[i][:][:].max()
#     bandRangePre = bandMaxPre-bandMinPre
#     for j in range(featuresPre.shape[1]):
#         for k in range(featuresPre.shape[2]):
#             featuresPre[i][j][k] = (featuresPre[i][j][k]-bandMinPre)/bandRangePre

# # Generating image chips from the array
# new_features = array_to_chips(featuresPre, x_size=7, y_size=7)

# #for i in range(len(models)):
#     # Predicting new data and exporting the probability raster
# newPredicted = model.predict(new_features)

# prediction = np.reshape(newPredicted.argmax(axis=1), (dsPre.RasterYSize, dsPre.RasterXSize))

# outFile = 'BrokenHill_ASTER_Oversampled_AlterationMap_CNN.tif'.format()

# raster.export(prediction, dsPre, filename=outFile, dtype='float')


















model = keras.models.load_model(r"D:\Internship\UNSW\Alteration_Zones\models_and_results\ASTER_MLP_Model\ASTER_MLP_Model")

# Loading and normalizing a new multispectral image
dsPre, featuresPre = raster.read('BrokenHill_ASTER.tif')
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
new_features.shape
#for i in range(len(models)):
    # Predicting new data and exporting the probability raster
# newPredicted = model.predict(new_features)

# prediction = np.reshape(newPredicted.argmax(axis=1), (dsPre.RasterYSize, dsPre.RasterXSize))

# outFile = 'BrokenHill_ASTER_MLP_AlterationMap_CNN.tif'.format()

# raster.export(prediction, dsPre, filename=outFile, dtype='float')