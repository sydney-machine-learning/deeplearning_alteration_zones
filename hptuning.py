#from Image_preprocessing import multiclass_preprocess
import time
import random
import numpy as np
import tensorflow as tf
from numpy import asarray
from numpy import save
from keras import backend as K
from numpy import load
import pandas as pd
import seaborn as sns
from pyrsgis import raster
from pyrsgis.ml import array_to_chips   #imagechipsfromarray doesnt work anymore.
from math import floor
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from numpy import load

NAME = "ASTER_DATA"     #Name of the folder in which the already preprocessed files exist.

train_x = load('Data/{}/train_x.npy'.format(NAME))
train_y = load('Data/{}/train_y.npy'.format(NAME))
validation_x = load('Data/{}/validation_x.npy'.format(NAME))
validation_y = load('Data/{}/validation_y.npy'.format(NAME))
test_x = load('Data/{}/test_x.npy'.format(NAME))
test_y = load('Data/{}/test_y.npy'.format(NAME))



#Function to collect the results on the test data
def results(model, test_x, test_y):
  y_pred = model.predict(test_x, batch_size=64, verbose=1)
  y_pred_final = np.empty_like(test_y)
  j = 0
  for i in y_pred:
    max_index = np.where(i == max(i))[0] 
    y_pred_final[j] = float(max_index) 
    j = j + 1


  Accuracy = accuracy_score(test_y, y_pred_final)
  precision = precision_score(test_y, y_pred_final, average = 'macro')
  recall = recall_score(test_y, y_pred_final, average = 'macro')
  F1_score = f1_score(test_y, y_pred_final, average = 'macro')

  return y_pred, Accuracy, precision, recall, F1_score

def calculateScoresFor30Runs(model, test_x, test_y, n_classes):
  fpr = {}
  tpr = {}
  roc_auc = {}

  y_pred = model.predict(test_x, batch_size=64, verbose=1)
  y_test_dummies = pd.get_dummies(test_y, drop_first=False).values
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  y_pred, Accuracy, precision, recall, F1_score = results(model, test_x, test_y)

  return Accuracy, precision, recall, F1_score, roc_auc

# import pandas as pd
# results_df = pd.read_csv("results_df.csv")
result_dict = {"ASTER_DATA_CNN_RMSPROP":[], "ASTER_DATA_CNN_ADAM":[], "ASTER_DATA_CNN_SGD":[], "ASTER_DATA_MLP":[], "LANDSAT8_DATA_CNN_RMSPROP":[], "LANDSAT8_DATA_CNN_ADAM":[], "LANDSAT8_DATA_CNN_SGD":[], "LANDSAT8_DATA_MLP":[]}

############## ORiginal Model Code ###################
for i in range(10):
    Model_name = "CNN_SGD"

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, padding='valid', activation='relu', input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, kernel_size= 3, padding='valid', activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))

    print(model.summary())
    #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    log_folder = "logs/{}".format(NAME)

    callbacks = [TensorBoard(log_dir=log_folder,
                    histogram_freq=1,
                      write_graph=True,
                      write_images=True,
                      update_freq='epoch',
                      profile_batch=2,
                      embeddings_freq=1)]

    model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs = 10, validation_data = (validation_x, validation_y))

    n_classes = 3
    Accuracy, precision, recall, F1_score, roc_auc = calculateScoresFor30Runs(model, test_x, test_y, n_classes)
    result_array = [Accuracy, precision, recall, F1_score]

    for i in range(n_classes):
      result_array.append(roc_auc[i])

    result_dict["{}_{}".format(NAME, Model_name)].append(result_array)



columns = ['Accuracy', 'precision', 'recall', 'F1_score', 'AUC_0', 'AUC_1', 'AUC_2']
result_array = np.array(result_dict["{}_{}".format(NAME, Model_name)])
result_array = np.transpose(result_array)
df = pd.DataFrame(result_array, columns)
df.to_csv("{}_{}.csv".format(NAME, Model_name))


reshaped_train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]*train_x.shape[2]*train_x.shape[3]))
reshaped_validation_x = np.reshape(validation_x, (validation_x.shape[0], validation_x.shape[1]*validation_x.shape[2]*validation_x.shape[3]))
reshaped_test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]*test_x.shape[2]*test_x.shape[3]))


for i in range(10):
    Model_name = "MLP"

    model = Sequential()


    model.add(Dense(64, activation='relu', input_shape=(train_x.shape[1]* train_x.shape[2]* train_x.shape[3],)))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    print(model.summary())
    #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    log_folder = "logs/{}".format(NAME)

    callbacks = [TensorBoard(log_dir=log_folder,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=1)]



    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    history = model.fit(reshaped_train_x, train_y, epochs = 10, validation_data = (reshaped_validation_x, validation_y))

    n_classes = 3
    Accuracy, precision, recall, F1_score, roc_auc = calculateScoresFor30Runs(model, test_x, test_y, n_classes)
    result_array = [Accuracy, precision, recall, F1_score]

    for i in range(n_classes):
      result_array.append(roc_auc[i])

    result_dict["{}_{}".format(NAME, Model_name)].append(result_array)

columns = ['Accuracy', 'precision', 'recall', 'F1_score', 'AUC_0', 'AUC_1', 'AUC_2']
result_array = np.array(result_dict["{}_{}".format(NAME, Model_name)])
result_array = np.transpose(result_array)
df = pd.DataFrame(result_array, columns)
df.to_csv("{}_{}.csv".format(NAME, Model_name))