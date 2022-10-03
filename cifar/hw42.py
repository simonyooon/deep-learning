# Simon Yoon
# ECE472 Deep Learning

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
)
from tensorflow.keras import regularizers, optimizers, layers, models, Sequential
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 32, "Size of batch in SGD")
flags.DEFINE_integer("epochs", 25, "Number of epochs")
flags.DEFINE_float("lr", 0.003, "Learning Rate")
flags.DEFINE_float("mom", 0.9, "Momentum")
flags.DEFINE_integer("group_size", 32, "Number of Groups in Norm")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_bool("debug", False, "Set logging level to debug")

WHICH = True


def load_cifar_dataset(shape=(-1, 32, 32, 3), path="hw4/data"):
    # helper fx @ https://www.cs.toronto.edu/~kriz/cifar.html
    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    # unpickle file and fill in data
    x_train = None
    y_train = []

    for i in range(1, 6):
        data_dic = unpickle(
            os.path.join(path, "cifar-10-batches-py/", "data_batch_{}".format(i))
        )
        if i == 1:
            x_train = data_dic[b"data"]
        else:
            x_train = np.vstack((x_train, data_dic[b"data"]))
        y_train += data_dic[b"labels"]

    test_data_dic = unpickle(os.path.join(path, "cifar-10-batches-py/", "test_batch"))

    x_test = test_data_dic[b"data"]
    y_test = np.array(test_data_dic[b"labels"])

    if shape == (-1, 3, 32, 32):
        x_test = x_test.reshape(shape)
        x_train = x_train.reshape(shape)
    elif shape == (-1, 32, 32, 3):
        x_test = x_test.reshape(shape, order="F")
        x_train = x_train.reshape(shape, order="F")
        x_test = np.transpose(x_test, (0, 2, 1, 3))
        x_train = np.transpose(x_train, (0, 2, 1, 3))
    else:
        x_test = x_test.reshape(shape)
        x_train = x_train.reshape(shape)

    y_train = np.array(y_train)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_cifar_dataset()


img_rows, img_cols, channels = 32, 32, 3
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()

datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    # zoom_range=0.3
)
datagen.fit(x_train)

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].astype(np.uint8))
    plt.show()
    break

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, 1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
mean = np.mean(x_train)
std = np.std(x_train)
x_test = (x_test - mean) / std
x_train = (x_train - mean) / std

# labels
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def plothist(hist):
    plt.plot(hist.history["acc"])
    plt.plot(hist.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


# build again, same model as model1

# reg=l2(1e-4)   # L2 or "ridge" regularization
reg2 = None
num_filters2 = 32
ac2 = "relu"
adm2 = Adam(lr=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = adm2
drop_dense2 = 0.5
drop_conv2 = 0

model = Sequential()

model.add(
    Conv2D(
        num_filters2,
        (3, 3),
        activation=ac2,
        kernel_regularizer=reg2,
        input_shape=(img_rows, img_cols, channels),
        padding="same",
    )
)
model.add(BatchNormalization(axis=-1))
model.add(
    Conv2D(
        num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, padding="same"
    )
)
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 16x16x3xnum_filters
model.add(Dropout(drop_conv2))

model.add(
    Conv2D(
        2 * num_filters2,
        (3, 3),
        activation=ac2,
        kernel_regularizer=reg2,
        padding="same",
    )
)
model.add(BatchNormalization(axis=-1))
model.add(
    Conv2D(
        2 * num_filters2,
        (3, 3),
        activation=ac2,
        kernel_regularizer=reg2,
        padding="same",
    )
)
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 8x8x3x(2*num_filters)
model.add(Dropout(drop_conv2))

model.add(
    Conv2D(
        4 * num_filters2,
        (3, 3),
        activation=ac2,
        kernel_regularizer=reg2,
        padding="same",
    )
)
model.add(BatchNormalization(axis=-1))
model.add(
    Conv2D(
        4 * num_filters2,
        (3, 3),
        activation=ac2,
        kernel_regularizer=reg2,
        padding="same",
    )
)
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 4x4x3x(4*num_filters)
model.add(Dropout(drop_conv2))

model.add(Flatten())
model.add(Dense(512, activation=ac2, kernel_regularizer=reg2))
model.add(BatchNormalization())
model.add(Dropout(drop_dense2))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=opt2
)

model.summary()

# train with image augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    steps_per_epoch=len(x_train) / 128,
    epochs=FLAGS.epochs,
    validation_data=(x_test, y_test),
)

# training accuracy without dropout
train_acc = model.evaluate(x_train, y_train, batch_size=128)
train_acc
plothist(history)

model_test_acc = model.evaluate(x_test, y_test, batch_size=128)
model_test_acc

model_train_acc = model.evaluate(x_train, y_train, batch_size=128)
model_train_acc
