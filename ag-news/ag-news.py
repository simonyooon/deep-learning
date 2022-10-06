# Simon Yoon
# ECE 472 Deep Learning
# Prof. Curro

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Reshape
from keras.layers import Activation
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras import regularizers

from sklearn.model_selection import train_test_split

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_classes", 4, "Number of classes")
flags.DEFINE_integer("num_epochs", 2, "Number of epochs") 
flags.DEFINE_integer("batch_size", 128, "Number of samples in batch")

#read in data
train_data = pd.read_csv('train.csv',header=None,names=['label','title','desc'])
test_data = pd.read_csv('test.csv',header=None,names=['label','title','desc'])
x_train = train_data['title'] + " " + train_data['desc']
y_train = train_data['label']
x_test = test_data['title'] + " " + test_data['desc']
y_test = test_data['label']
x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=.1, random_state=123)

#transform data to hashed integer sequences
vocabulary = 20000
keep = int(vocabulary*.001)
e_train = [one_hot(d,vocabulary) for d in x_train]
e_val = [one_hot(d,vocabulary) for d in x_val]
e_test = [one_hot(d,vocabulary) for d in x_test]

max_length = 75
p_train = np.array(pad_sequences(e_train,maxlen=max_length,padding='post'))
p_val = np.array(pad_sequences(e_val,maxlen=max_length,padding='post'))
p_test = np.array(pad_sequences(e_test,maxlen=max_length,padding='post'))

trainLen = p_train.shape[0]
valLen = p_val.shape[0]
testLen = p_test.shape[0]

p_train = p_train.flatten()
p_val = p_val.flatten()
p_test = p_test.flatten()
trainBin = np.bincount(p_train)
trainTops = [trainBin[i] for i in trainBin.argsort()[:-keep]]

p_train = p_train.reshape(trainLen,max_length)
p_val = p_val.reshape(valLen,max_length)
p_test = p_test.reshape(testLen,max_length)

y_train[y_train==4] = 0
y_val[y_val==4] = 0
y_test[y_test==4] = 0
y_train = keras.utils.to_categorical(y_train, FLAGS.num_classes)
y_val = keras.utils.to_categorical(y_val, FLAGS.num_classes)
y_test = keras.utils.to_categorical(y_test, FLAGS.num_classes)

model = Sequential()
model.add(Embedding(vocabulary, 4, input_length=max_length,name='embed'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(FLAGS.num_classes, activation='softmax'))

optimizer = keras.optimizers.Adam()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(p_train, y_train,
          batch_size=FLAGS.batch_size,
          epochs=FLAGS.num_epochs,
          verbose=1,
          validation_data=(p_val, y_val))
score = model.evaluate(p_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])