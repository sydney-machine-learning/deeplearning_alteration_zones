from Image_preprocessing import multiclass_preprocess
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import time
import tensorflow as tf
from numpy import asarray
from numpy import save
from keras import backend as K
from numpy import load
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Defining file names

feature_file = 'BrokenHill_Landsat8.tif'
positiveLabel_file = 'BrokenHill_Landsat8_CombinedTraining_PCA.tif'

############# All the preprocessing of the image has been done in the image_preprocessing.py file ##############
multiclass_preprocess_instance = multiclass_preprocess(feature_file, positiveLabel_file)
train_x, train_y, test_x, test_y = multiclass_preprocess_instance.train_test_split()


#save numpy array as csv file

save('train_x.npy', train_x)
save('train_y.npy', train_y)
save('test_x.npy', test_x)
save('test_y.npy', test_y)

train_x = load('train_x.npy')
train_y = load('train_y.npy')
test_x = load('test_x.npy')
test_y = load('test_y.npy')

# #FOR ASTER DATA:
# print(sum(train_y == 0))   #147018
# print(sum(train_y == 1))   #80691
# print(sum(train_y == 2))   #213631
# print(sum(train_y == 3))

#FOR LANDSAT8 DATA:
print(sum(train_y == 0))   #24136
print(sum(train_y == 1))   #15340
print(sum(train_y == 2))   #33122
print(sum(train_y == 3))


# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))



# results_df = pd.DataFrame(columns = ['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])

# ########### Original Model ###########
# model = Sequential()

# model.add(Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))
# model.add(Dropout(0.25))
# model.add(Conv2D(48, kernel_size= 1, padding='valid', activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))

# print(model.summary())
# #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# NAME = "Original_Model"
# log_folder = "logs/{}".format(NAME)

# callbacks = [TensorBoard(log_dir=log_folder,
#                 histogram_freq=1,
#                   write_graph=True,
#                   write_images=True,
#                   update_freq='epoch',
#                   profile_batch=2,
#                   embeddings_freq=1)]

# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1_m, precision_m, recall_m])
# model.fit(train_x, train_y, epochs = 10, validation_data = (test_x, test_y), callbacks = callbacks)
# y_pred = model.predict(test_x, batch_size=64, verbose=1)

# y_pred_final = np.empty_like(test_y)
# j = 0
# for i in y_pred:
#   max_index = np.where(i == max(i))[0] 
#   y_pred_final[j] = float(max_index) 
#   j = j + 1


# Accuracy = accuracy_score(test_y, y_pred_final)
# precision = precision_score(test_y, y_pred_final, average = 'macro')
# recall = recall_score(test_y, y_pred_final, average = 'macro')
# F1_score = f1_score(test_y, y_pred_final, average = 'macro')

# results_df = results_df.append({'Model' : 'Original_Model', 'Accuracy' : Accuracy, 'F1-Score': F1_score, 'Precision': precision, 'Recall': recall }, ignore_index = True)
# print(results_df)

# model.save('models/Original_model')










# ############# Code for testing multiple Model Architectures ##############

# # dense_layers = [0, 1, 2]
# # kernel_sizes = [1, 2, 3]
# # #layer_sizes = [32, 48, 64]
# # conv_layers = [2]


# #Dense layers kept the same
# #Kernel Sizes in different Convolutional layers changed 
# #Number of Convolution layers changed
# #Layer size in the convolution layers also changed
# # for kernel_size in kernel_sizes:
# # #    for layer_size in layer_sizes:
# #     for conv_layer in conv_layers:
# #         NAME = "{}-conv-{}-kernel-{}".format(conv_layer, kernel_size, int(time.time()))
# #         print(NAME)

# #         model = Sequential()

# #         model.add(Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))
# #         model.add(Dropout(0.25))

# #         for l in range(conv_layer-1):
# #             model.add(Conv2D(48, kernel_size= kernel_size))
# #             model.add(Activation('relu'))
# #             model.add(Dropout(0.25))

# #         model.add(Flatten())

# #         model.add(Dense(64, activation='relu'))
# #         model.add(Dropout(0.5))
# #         model.add(Dense(4, activation='softmax'))

# #         #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# #         log_folder = "logs/{}".format(NAME)
# #         callbacks = [TensorBoard(log_dir=log_folder,
# #                         histogram_freq=1,
# #                          write_graph=True,
# #                          write_images=True,
# #                          update_freq='epoch',
# #                          profile_batch=2,
# #                          embeddings_freq=1)]

# #         print(model.summary())

# #         # Running the model
# #         model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# #         model.fit(train_x, train_y, epochs=2, validation_data = (test_x, test_y), callbacks = callbacks)





