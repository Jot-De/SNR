import os
import numpy as np
import matplotlib.pyplot as plt

from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential, load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau, \
    LearningRateScheduler
from keras.metrics import sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
from keras import layers
from keras.layers.advanced_activations import ReLU

# os.chdir("C:\\Users\\Janek\\Desktop\\EITI\\SNR\\klasyfikacja psów\\code")
# os.chdir("C:\\Users\\Piotr\\Documents\\Studia\\Informatyka PW\\2 semestr\\SNR\\Projekt")
os.chdir("C:\\Users\\Marcin  Piotrek\\Desktop\\SNR\\Projekt")
# image_dir = '../input/images/dataset/'
# image_dir = '../input/images/Images_3'
train_dir = '../input/images/dataset/train/'
val_dir = '../input/images/dataset/val/'
test_dir = '../input/images/dataset/test/'

imgsize = 224
batch_size = 4
epochs = 50
classes_number = 120
train_samples_number = 16418
val_samples_number = 2009
test_samples_number = 2153


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


def setup():
    # Data generators - rescale to change pixel values from 0-255 to 0-1
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generate batches of images on demand from directories
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(imgsize, imgsize), class_mode='sparse',
                                                        batch_size=batch_size)
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(imgsize, imgsize), class_mode='sparse',
                                                    batch_size=batch_size, shuffle=False)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(imgsize, imgsize), class_mode='sparse',
                                                      batch_size=1, shuffle=False)

    return train_generator, val_generator, test_generator


def create_model():
    base_model = load_model("model_3.h5")
    # the output layer is 'block_16_depthwise_relu
    layers_to_del = ['input_1', 'block_15_expand', 'block_15_expand_relu', 'block_15_depthwise',
                     'block_15_depthwise_relu' 'block_15_project', 'block_15_project_BN']
    model = layers.Input(shape=(imgsize, imgsize, 3))
    for mobilenet_layer in base_model.layers[0].layers:
        if mobilenet_layer.name not in layers_to_del:
            model = base_model.layers[0].get_layer(mobilenet_layer.name)(model)
    model = base_model.get_layer(1)(model)
    model = base_model.get_layer(2)(model)
    # base_model.layers[0].layers.pop()
    # base_model.layers[0].layers.pop()
    # base_model.layers[0].layers.pop()
    # base_model.layers[0].layers.pop()
    # base_model.layers[0].layers.pop()
    # base_model.layers.pop()
    # base_model.layers.pop()
    # # Classification head
    # pooling = layers.GlobalAveragePooling2D()(base_model.layers[0].get_layer('block_16_depthwise_relu').output)
    # out = layers.Dense(classes_number, activation='softmax', use_bias=True, name='Logits')(pooling)
    # model = Model(input=base_model.layers[0].get_input_at(0), output=[out])

    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    # plot_model(model, to_file='model_plot_3b.png', show_shapes=True, show_layer_names=True)
    # -----------Optimizers-----------#
    opt1 = SGD(lr=1e-4, momentum=0.99)
    opt = Adam(lr=1e-4)
    # ----------Compile---------------#
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=[sparse_categorical_accuracy, sparse_top_k_categorical_accuracy]
    )

    # model.set_weights(base_model.get_weights())

    model.summary()
    return model


def train(model, train_generator, val_generator):
    # Save to file learning data after each epoch
    checkpoint = ModelCheckpoint(
        './base.model_3b',
        monitor='val_sparse_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        period=1
    )
    # Stop training when a monitored quantity has stopped improving.
    earlystop = EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        min_delta=0.1,
        patience=20,
        verbose=1,
        mode='auto'
    )
    # Log history of teaching
    tensorboard = TensorBoard(
        log_dir='./logs_3b',
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=True,
        write_images=False,
    )
    # Logs of learning
    csvlogger = CSVLogger(
        filename="training_csv_3b.log",
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
    # -----------Training------------#
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples_number / batch_size,
        validation_data=val_generator,
        validation_steps=val_samples_number / batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks
    )
    show_final_history(history)


def evaluate(model, test_generator):
    model.load_weights('./base.model_3b')
    # Reset generator for correct samples order
    test_generator.reset()
    # Test generator gives one sample per step
    model_score = model.evaluate_generator(test_generator, steps=test_samples_number)
    print("Model Test Loss:", model_score[0])
    print("Model Test Accuracy:", model_score[1])
    print("Model Test Top-5 Accuracy", model_score[2])

    test_generator.reset()
    predictions = model.predict_generator(test_generator, steps=test_samples_number)
    predictions_labels = np.argmax(predictions, axis=1)
    with open("confusion_matrix_3b.csv", "w") as file:
        np.savetxt(file, confusion_matrix(test_generator.classes, y_pred=predictions_labels), delimiter=",")
    print('Classification Report')
    target_names = test_generator.class_indices.keys()
    print(classification_report(test_generator.classes, predictions_labels, target_names=target_names))

    model_json = model.to_json()
    with open("model_3b.json", "w") as json_file:
        json_file.write(model_json)

    model.save("model_3b.h5")
    print("Weights Saved")


if __name__ == '__main__':
    train_generator, val_generator, test_generator = setup()
    model = create_model()
    train(model, train_generator, val_generator)
    evaluate(model, test_generator)
