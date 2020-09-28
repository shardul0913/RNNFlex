import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv1D,Conv2D, MaxPool1D, MaxPool2D,Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def Load():
    stars = pd.read_csv('exoTrain.csv')
    outStars = pd.read_csv('exoTest.csv')

    y_train = stars.loc[:,'LABEL':]
    x_train = stars.loc[:,'FLUX.1':]

    y_test = outStars.loc[:,'LABEL':]
    x_test = outStars.loc[:,'FLUX.1':]

    scaler = MinMaxScaler()

    y_train = scaler.fit_transform(y_train)
    x_train = scaler.fit_transform(x_train)

    print(x_train.shape[1:])
    print(x_train.shape)
    return x_train,y_train

def CNNtrain():
    x_train,y_train = Load()
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

    model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])

Load()
CNNtrain()