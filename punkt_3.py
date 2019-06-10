import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc

from tqdm import tqdm
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
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

base_model = MobileNetV2(input_shape=(imgsize, imgsize, 3), weights='imagenet', include_top=False, classes=len(dirs))
# Create own classifier head
model = Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(len(dirs), activation='softmax', use_bias=True, name='Logits'))
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot_3.png', show_shapes=True, show_layer_names=True)

# Save to file learning data after each epoch
checkpoint = ModelCheckpoint(
    './base.model_3',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
# Stop training when a monitored quantity has stopped improving.
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
# Log history of teaching
tensorboard = TensorBoard(
    log_dir='./logs_3',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)
# Logs of learning
csvlogger = CSVLogger(
    filename="training_csv_3.log",
    separator=",",
    append=False
)
# Reduce learning rate when a metric has stopped improving.
reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1,
    mode='auto'
)

callbacks = [checkpoint, tensorboard, csvlogger, reduce, earlystop]

# -----------Optimizers-----------#
opt1 = SGD(lr=1e-4, momentum=0.99)
opt = Adam(lr=1e-2)
# ----------Compile---------------#
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
# -----------Training------------#
history = model.fit_generator(
    augs_gen.flow(x_train, y_train, batch_size=16),
    validation_data=augs_gen.flow(x_val, y_val, batch_size=len(x_val)),
    steps_per_epoch=ceil(len(x_train) / 16),
    validation_steps=ceil(len(x_val) / 32),
    epochs=3,
    verbose=2,
    callbacks=callbacks
)

show_final_history(history)
model.load_weights('./base.model_3')
model_score = model.evaluate_generator(augs_gen.flow(x_test, y_test, batch_size=32), steps=ceil(len(x_test) / 32))
print("Model Test Loss:", model_score[0])
print("Model Test Accuracy:", model_score[1])

model_json = model.to_json()
with open("model_3.json", "w") as json_file:
    json_file.write(model_json)

model.save("model_3.h5")
print("Weights Saved")
