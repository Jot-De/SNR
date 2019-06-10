import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

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
# image_dir = '../input/images/dataset/'
# image_dir = '../input/images/Images_3'
train_dir = '../input/images/dataset/train/'
val_dir = '../input/images/dataset/val/'
test_dir = '../input/images/dataset/test/'

imgsize = 224
batch_size = 16
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
                                                    batch_size=batch_size)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(imgsize, imgsize), class_mode='sparse',
                                                      batch_size=batch_size)

    return train_generator, val_generator, test_generator


def create_model():
    # Load pretrained model
    base_model = MobileNetV2(input_shape=(imgsize, imgsize, 3), weights='imagenet', include_top=False,
                             classes=classes_number)
    # Conv_1 is the name of the last convolutional layer
    for layer in base_model.layers:
        if layer.get_config()['name'] == 'Conv_1':
            layer.trainable = True
        else:
            layer.trainable = False
    # Create own classifier head
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(classes_number, activation='softmax', use_bias=True, name='Logits'))

    # Train classifying layer
    layers_to_train = ['Logits']

    for layer in model.layers:
        if layer.get_config()['name'] in layers_to_train:
            layer.trainable = True

    # Print model summaries
    model.summary()
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True)

    return model


def train(model, train_generator, val_generator):
    # Callbacks
    # Save to file learning data after each epoch
    checkpoint = ModelCheckpoint(
        './base.model_2',
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
        min_delta=0.01,
        patience=20,
        verbose=1,
        mode='auto'
    )
    # Log history of teaching
    tensorboard = TensorBoard(
        log_dir='./logs_2',
        histogram_freq=0,
        batch_size=16,
        write_graph=True,
        write_grads=True,
        write_images=False,
    )
    # Logs of learning
    csvlogger = CSVLogger(
        filename="training_csv_2.log",
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
    opt = Adam(lr=1e-3)
    # ----------Compile---------------#
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
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
    model.load_weights('./base.model_2')
    model_score = model.evaluate_generator(test_generator, steps=test_samples_number / batch_size)
    print("Model Test Loss:", model_score[0])
    print("Model Test Accuracy:", model_score[1])

    predictions = model.predict_generator(test_generator, steps=test_samples_number / batch_size)
    predictions_labels = np.argmax(predictions, axis=1)
    print("Confusion matrix")
    print(confusion_matrix(test_generator.classes, y_pred=predictions_labels))
    print('Classification Report')
    target_names = test_generator.class_indices.keys()
    print(classification_report(test_generator.classes, predictions_labels, target_names=target_names))

    model_json = model.to_json()
    with open("model_2.json", "w") as json_file:
        json_file.write(model_json)

    model.save("model_2.h5")
    print("Weights Saved")


if __name__ == '__main__':
    train_generator, val_generator, test_generator = setup()
    model = create_model()
    train(model, train_generator, val_generator)
    evaluate(model, test_generator)
