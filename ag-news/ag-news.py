# Simon Yoon
# ECE 472 Deep Learning
# Prof. Curro
# AG-News Classification

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.preprocessing.text import one_hot
from keras.layers import Reshape, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras import regularizers

from sklearn.model_selection import train_test_split

from absl import app, flags

"""
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_classes", 4, "Number of classes")
flags.DEFINE_integer("num_epochs", 2, "Number of epochs") 
flags.DEFINE_integer("batch_size", 128, "Number of samples in batch")
flags.DEFINE_integer("max_max_lenth", 75)
"""

num_classes = 4
num_epochs = 2
batch_size = 128

# read in data, process
train_data = pd.read_csv("train.csv", header=None, names=["label", "title", "desc"])
test_data = pd.read_csv("test.csv", header=None, names=["label", "title", "desc"])
x_train = train_data["title"] + " " + train_data["desc"]
y_train = train_data["label"]
x_test = test_data["title"] + " " + test_data["desc"]
y_test = test_data["label"]

# split training data - cross validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

# data to hashed integer sequences, one hot encoding vs. tokenizer
vocabulary = 20000

e_x_train = [one_hot(d, vocabulary) for d in x_train]
e_x_valid = [one_hot(d, vocabulary) for d in x_valid]
e_x_test = [one_hot(d, vocabulary) for d in x_test]

max_len = int(x_train.str.len().quantile(0.95))
p_train = np.array(pad_sequences(e_x_train, maxlen=max_len, padding="post"))
p_valid = np.array(pad_sequences(e_x_valid, maxlen=max_len, padding="post"))
p_test = np.array(pad_sequences(e_x_test, maxlen=max_len, padding="post"))

train_size = p_train.shape[0]
valid_size = p_valid.shape[0]
test_size = p_test.shape[0]

p_train = p_train.flatten()
p_valid = p_valid.flatten()
p_test = p_test.flatten()

p_train = p_train.reshape(train_size, max_len)
p_valid = p_valid.reshape(valid_size, max_len)
p_test = p_test.reshape(test_size, max_len)

y_train[y_train == 4] = 0
y_valid[y_valid == 4] = 0
y_test[y_test == 4] = 0

# convert vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Embedding(vocabulary, 4, input_length=max_len))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

model.fit(
    p_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=1,
    validation_data=(p_valid, y_valid),
)
score = model.evaluate(p_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
