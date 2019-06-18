# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:54:38 2019

@author: Janek
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau, \
    LearningRateScheduler
from keras import layers

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
import numpy as np
from keras import metrics
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pandas as pd
import sklearn.metrics as skm

img_width = 224
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 1
class_nb = 120

os.chdir("C:\\Users\\Janek\\Desktop\\EITI\\SNR\\klasyfikacja\\code")
#os.chdir("C:\\Users\\Piotr\\Documents\\Studia\\Informatyka PW\\2 semestr\\SNR\\Projekt")
# image_dir = '../input/images/Images/'
image_test_dir = '../input/images/Images_3'
image_train_dir = '../input/images/Images_3'

final_model= load_model("model_3b.h5")


print(final_model.summary())


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)

train_set = train_datagen.flow_from_directory(image_train_dir,
                                              target_size=(img_width, img_width),
                                              batch_size=batch_size,
                                              class_mode='sparse')

valid_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)

valid_set = valid_datagen.flow_from_directory(image_test_dir,
                                              target_size=(img_width, img_width),
                                              batch_size=batch_size,
                                              class_mode='sparse')
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(15780):
    _x, _y = train_set.next()
    _x_embed = final_model.predict(_x[:, :, :])
    X_train.append(_x_embed[0, :])
    y_train.append(_y[0, :])
    if i % 100 == 0:
        print(i)

for i in range(3600):
    _x, _y = valid_set.next()
    _x_embed = final_model.predict(_x[:, :, :])
    X_test.append(_x_embed[0, :])
    y_test.append(_y[0, :])
    if i % 100 == 0:
        print(i)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train_df = pd.DataFrame(X_train)
X_train_df.to_csv('SVM_4b_X_train.csv', index=False)

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv('SVM_4b_y_train.csv', index=False)

X_test_df = pd.DataFrame(X_test)
X_test_df.to_csv('SVM_4b_X_test.csv', index=False)

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv('SVM_4b_y_test.csv', index=False)