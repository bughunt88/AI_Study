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

import warnings

warnings.filterwarnings('ignore')

#trainset = np.load('../data/project/data/train_data3.npy',allow_pickle=True)
#testset = np.load('../data/project/data/test_data3.npy',allow_pickle=True)

trainset = np.load('../data/project/data/kfold_data3.npy',allow_pickle=True)


# split each set into raw data, mfcc data, and y data
# STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel0spectogram 한 것
train_X = []
train_mfccs = []
train_y = []

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

train_mfccs = [a for (a,b) in trainset]
train_y = [b for (a,b) in trainset]

train_mfccs = np.array(train_mfccs)
train_y = to_categorical(np.array(train_y))

train_X_ex = np.expand_dims(train_mfccs, -1)
print('train X shape:', train_X_ex.shape)

print(train_X_ex[0].shape)

x_train, x_test, y_train, y_test = train_test_split(train_X_ex, train_y,  train_size=0.9, random_state = 66 ) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 66 ) 

ip = Input(shape=train_X_ex[0].shape)
m = Conv2D(64, kernel_size=(2,2), activation='relu')(ip)
m = BatchNormalization(axis=-1)(m)
m = Conv2D(64, kernel_size=(2,2), activation='relu')(m)
m = MaxPooling2D(pool_size=(4,4))(m)
m = BatchNormalization(axis=-1)(m)
m = Dropout(0.3)(m)
m = Conv2D(64, kernel_size=(2,2), activation='relu')(m)
m = BatchNormalization(axis=-1)(m)
m = Conv2D(64, kernel_size=(4,4), activation='relu')(m)
m = BatchNormalization(axis=-1)(m)
m = Dropout(0.3)(m)

m = Flatten()(m)

m = Dense(1024, activation='relu')(m)
m = BatchNormalization(axis=-1)(m)
m = Dropout(0.3)(m)
m = Dense(512, activation='relu')(m)
m = BatchNormalization(axis=-1)(m)
op = Dense(3, activation='softmax')(m)

model = Model(ip, op)

model.summary()



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


eraly_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') # mode는 min,max,auto 있다
reLR = ReduceLROnPlateau(patience=5,verbose=1,factor=0.5) #learning rate scheduler

history = model.fit(x_train,
                    y_train,
                    epochs=500,
                    batch_size=8,
                    verbose=1,
                    validation_data=(x_val, y_val), callbacks=[eraly_stopping,reLR])


model.save('../data/project/file/save_model.h5')

loss, acc= model.evaluate(x_test, y_test, batch_size=8)
# 지표를 만들기 위한 프레딕트 
print('loss : ',loss)
print('acc : ',acc)


DATA_DIR = '../data/project/predict/'

for filename in os.listdir(DATA_DIR):
    filename = normalize('NFC', filename)

    wav, sr = librosa.load(DATA_DIR + filename,sr=16000)
    mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=50, n_fft=400, hop_length=160)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    padded_mfcc = pad2d(mfcc, 650)
    
    padded_mfcc= np.expand_dims(padded_mfcc, 0)

    y_pred = model.predict(padded_mfcc)
    
    y_predict=np.argmax(y_pred, axis=1)

    if y_predict == 0:
        y_predict = '분노'
    elif y_predict == 1:
        y_predict = '평상시'
    elif y_predict == 2:
        y_predict = '슬픔'


    print('파일 명 : ',filename)
    print('예측값 : ', y_predict)

# 101~150 평상시 - 1
# 251~300 분노 - 0
# 301~350 슬픔 - 2


'''
loss :  1.6047866344451904
acc :  0.7803468108177185
'''