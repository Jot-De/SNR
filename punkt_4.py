import os
import numpy as np
import matplotlib.pyplot as plt

from math import ceil

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

# os.chdir("C:\\Users\\Janek\\Desktop\\EITI\\SNR\\klasyfikacja psów\\code")
os.chdir("C:\\Users\\Piotr\\Documents\\Studia\\Informatyka PW\\2 semestr\\SNR\\Projekt")
# image_dir = '../input/images/Images/'
image_dir = '../input/images/Images_3'

dirs = os.listdir(image_dir)

X = []
Z = []
imgsize = 224


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


def label_assignment(img, label):
    return label


def training_data(label, data_dir):
    """
    Load data and puts into an array with label
    :param label:
    :param data_dir:
    :return:
    """
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img, label)
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))

        X.append(np.array(img))
        Z.append(str(label))


for dir_name in dirs:
    """
    Iterates through directories
    """
    full_dir = os.path.join(image_dir, dir_name)
    label = dir_name.split(sep='-', maxsplit=1)[1]
    training_data(label, full_dir)

X = np.array(X)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Z)
del Z

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=69)
del X
del Y
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=13)
# Rescale to change pixel values from 0-255 to 0-1
augs_gen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    zoom_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    vertical_flip=False)

augs_gen.fit(x_train)

model = load_model("model_3.h5")
# layers_to_del = ['Conv_1', 'Conv_1_bn', 'out_relu']
model.pop()

model.summary()
