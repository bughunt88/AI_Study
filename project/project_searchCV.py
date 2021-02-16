
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import librosa
import sklearn
import random
from unicodedata import normalize
from keras.layers import Dense
from keras import Input
from keras.engine import Model
from keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

# 1. 데이터 / 전처리

import warnings

warnings.filterwarnings('ignore')

trainset = np.load('../data/project/data/train_data.npy',allow_pickle=True)
testset = np.load('../data/project/data/test_data.npy',allow_pickle=True)

# split each set into raw data, mfcc data, and y data
# STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel0spectogram 한 것
train_X = []
train_mfccs = []
train_y = []

test_X = []
test_mfccs = []
test_y = []

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

train_mfccs = [a for (a,b) in trainset]
train_y = [b for (a,b) in trainset]

test_mfccs = [a for (a,b) in testset]
test_y = [b for (a,b) in testset]

train_mfccs = np.array(train_mfccs)
train_y = to_categorical(np.array(train_y))

test_mfccs = np.array(test_mfccs)
test_y = to_categorical(np.array(test_y))

print('train_mfccs:', train_mfccs.shape)
print('train_y:', train_y.shape)

print('test_mfccs:', test_mfccs.shape)
print('test_y:', test_y.shape)

train_X_ex = np.expand_dims(train_mfccs, -1)
test_X_ex = np.expand_dims(test_mfccs, -1)
print('train X shape:', train_X_ex.shape)
print('test X shape:', test_X_ex.shape)

x_train, x_val, y_train, y_val = train_test_split(train_X_ex, train_y,  train_size=0.8, random_state = 66 ) 


# 2. 모델
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def build_model(drop=0.5, optimizer='adam', activation='relu', kernel='2', pool='2') :

        
    ip = Input(shape=train_X_ex[0].shape, name='input')

    m = Conv2D(32, kernel_size=(kernel), activation=activation)(ip)
    m = MaxPooling2D(pool_size=(pool))(m)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(32*2, kernel_size=(kernel), activation=activation)(ip)
    m = MaxPooling2D(pool_size=(pool))(m)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(32*3, kernel_size=(kernel), activation=activation)(ip)
    m = MaxPooling2D(pool_size=(pool))(m)
    m = BatchNormalization(axis=-1)(m)

    m = Flatten()(m)

    m = Dense(64, activation=activation)(m)

    m = Dense(32, activation=activation)(m)

    op = Dense(3, activation='softmax')(m)
    model = Model(ip, op)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model

def create_hyperparameters() :
    batches = [2, 4, 8, 16]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    activation =['relu','elu']
    kernel = [2,3,4]
    pool = [2,3,4]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "activation" : activation, "kernel" : kernel, "pool" : pool}

model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=100)   

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=5, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model2, hyperparameters, cv=2)

search.fit(x_train, y_train, verbose=1, validation_split=0.2, callbacks=[es, lr],validation_data=(x_val, y_val))

print("best_params : ", search.best_params_)         
print("best_estimator : ", search.best_estimator_)   
print("best_score : ", search.best_score_)           