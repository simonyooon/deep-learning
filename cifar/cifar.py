# Simon Yoon
# ECE472 Deep Learning
# Professor Curro

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

"""
Developing a model with state-of-the-art performance was difficult with limited CPU performance. 
For example, training with a model analagous to Google's Inception took hours for a few epochs. 
Attempted scheduled learning rate changes, regularization and initialization schemes, weight standardization,
and experimentation with convolutional/dense layers with varying success in combined usage.
This model worked fast enough for me to test different things in a modular fashion. The most promising change
was data augmentation and retraining. CIFAR10 with this model reached a validation accuracy of ~ 88% at 
50 epochs and ~ 82% at ~ 40 epochs for CIFAR100.
"""
# data
# helper fx @ https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_data(test_path, train_path, labels_path, cifar10):
    data_test = unpickle(test_path)
    data_labels = unpickle(labels_path)
    x_test = (
        data_test[b"data"]
        .reshape(len(data_test[b"data"]), 3, 32, 32)
        .transpose(0, 2, 3, 1)
    )

    y_test_key = b"labels" if cifar10 else b"fine_labels"
    y_test = np.array(data_test[y_test_key])

    labels_key = b"label_names" if cifar10 else b"fine_label_names"
    labels = np.array(data_labels[labels_key])

    x_train = []
    y_train = []
    if cifar10:
        for i in range(1, 6):
            data_train = unpickle(f"{train_path}{i}")
            x_train.append(
                data_train[b"data"]
                .reshape(len(data_train[b"data"]), 3, 32, 32)
                .transpose(0, 2, 3, 1)
            )
            y_train.append(data_train[b"labels"])
    else:
        data_train = unpickle(train_path)
        x_train.append(
            data_train[b"data"]
            .reshape(len(data_train[b"data"]), 3, 32, 32)
            .transpose(0, 2, 3, 1)
        )
        y_train.append(data_train[b"fine_labels"])

    x_train = np.array(x_train).reshape(50000, 32, 32, 3)
    y_train = np.array(y_train).flatten()
    return x_test, x_train, y_test, y_train, labels


x_test, x_train, y_test, y_train, labels = get_data(
    "./cifar/data/cifar-10-batches-py/test_batch",
    "./cifar/data/cifar-10-batches-py/data_batch_",
    "./cifar/data/cifar-10-batches-py/batches.meta",
    True,
)
x_test_100, x_train_100, y_test_100, y_train_100, labels_100 = get_data(
    "./cifar/data/cifar-100-python/test",
    "./cifar/data/cifar-100-python/train",
    "./cifar/data/cifar-100-python/meta",
    False,
)

# Cross-Validation
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

x_train_100 = x_train_100 / 255.0
x_test_100 = x_test_100 / 255.0
x_train_100, x_valid_100, y_train_100, y_valid_100 = train_test_split(
    x_train_100, y_train_100, test_size=0.2
)

# visualize data by plotting images
fig, ax = plt.subplots(5, 5)
k = 0

for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect="auto")
        k += 1

plt.show()

# number of classes
K = len(set(y_train))  # 10, 100
# Model for CIFAR-10
# https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
# Input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(
    32,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(i)
x = BatchNormalization()(x)
x = Conv2D(
    32,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(
    64,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = Conv2D(
    64,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(
    128,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = Conv2D(
    128,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)

# Output layer
x = Dense(K, activation="softmax")(x)

model = Model(i, x)

# Model description
model.summary()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Fit
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)


# Fit with data augmentation
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit(
    train_generator,
    validation_data=(x_valid, y_valid),
    steps_per_epoch=steps_per_epoch,
    epochs=25,
)


# Plot accuracy per epoch
plt.plot(r.history["accuracy"], label="acc", color="red")
plt.plot(r.history["val_accuracy"], label="val_acc", color="green")
plt.legend()


# number of classes
K = 100  # 10, 100
# Model for CIFAR-100
# Input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(
    32,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(i)
x = BatchNormalization()(x)
x = Conv2D(
    32,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(
    64,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = Conv2D(
    64,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(
    128,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = Conv2D(
    128,
    (3, 3),
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    padding="same",
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)

# Output layer
x = Dense(K, activation="softmax")(x)

model_100 = Model(i, x)

# Model description
model_100.summary()

# Compile
# https://keras.io/api/optimizers/adam/
model_100.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
)

# Fit
r_100 = model_100.fit(
    x_train_100, y_train_100, validation_data=(x_test_100, y_test_100), epochs=50
)


# Fit with data augmentation
train_generator_100 = data_generator.flow(x_train_100, y_train_100, batch_size)
steps_per_epoch_100 = x_train_100.shape[0] // batch_size

r_100 = model_100.fit(
    train_generator_100,
    validation_data=(x_valid_100, y_valid_100),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
)


# Plot accuracy per epoch
plt.plot(r.history["accuracy"], label="acc", color="red")
plt.plot(r.history["val_accuracy"], label="val_acc", color="green")
plt.legend()
