import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv1D,Conv2D, MaxPool1D, InputLayer, MaxPool2D,Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from keras import optimizers
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
from flask_restful import Api

from keras.applications import vgg16
from keras.models import Model

graph = tf.get_default_graph()


# app = Flask(__name__)
# api = Api(app)
# print('Visit http://127.0.0.1:5000')

def Load():
    stars = pd.read_csv('exoTrain.csv')
    outStars = pd.read_csv('exoTest.csv')

    stars = stars.to_numpy()
    x_train = stars[:, 1:]
    y_train = stars[:, 0, np.newaxis] - 1.

    outStars = outStars.to_numpy()

    x_test = outStars[:, 1:]
    y_test = outStars[:, 0, np.newaxis] - 1.
    #
    # y_train = stars.loc[:,'LABEL':]
    # x_train = stars.loc[:,'FLUX.1':]
    #
    # y_test = outStars.loc[:,'LABEL':]
    # x_test = outStars.loc[:,'FLUX.1':]
    #
    # scaler = MinMaxScaler()

    print(x_train.shape[1:])
    print(x_test.shape)

    x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1, 1)) /
               np.std(x_train, axis=1).reshape(-1, 1))
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1, 1)) /
              np.std(x_test, axis=1).reshape(-1, 1))

    ## expand the data to have 3 dimensions, 3rd dimension is dummy for the conv1d to accept
    x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
    x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)

    print(x_train.shape[1:])
    print(x_test.shape)

    return x_train,y_train,x_test,y_test


def batch_generator(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

    yes_idx = np.where(y_train[:, 0] == 1.)[0]
    non_idx = np.where(y_train[:, 0] == 0.)[0]

    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)

        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis=0)

        yield x_batch, y_batch

def CNNtrain():

    x_train,y_train,x_test,y_test = Load()
    n_timestamps = x_train.shape[0]
    n_features = x_train.shape[1]

    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Start with a slightly lower learning rate, to ensure convergence
    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit_generator(batch_generator(x_train, y_train, 32),
                               validation_data=(x_test, y_test),
                               verbose=0, epochs=5,
                               steps_per_epoch=x_train.shape[1] // 32)

def VGG16():

    x_train,y_train,x_test,y_test = Load()
    n_timestamps = x_train.shape[0]
    n_features = x_train.shape[1]

    vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                      input_shape=x_train.shape)

    ### Removed the last fully connected neural networks from VGG
    output = vgg.layers[-1].output
    vgg_model = Model(vgg.input, output)

    input_shape = vgg_model.output_shape[1]

    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    model.summary()

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    # Load()
    VGG16()
    # CNNtrain()