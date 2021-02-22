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
from sklearn.model_selection import KFold
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D, Activation, Add, GlobalAveragePooling2D

import warnings

warnings.filterwarnings('ignore')

kfoldset = np.load('../data/project/data/kfold_data1.npy',allow_pickle=True)

# split each set into raw data, mfcc data, and y data
# STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel0spectogram 한 것
train_X = []
train_mfccs = []
train_y = []

test_X = []
test_mfccs = []
test_y = []

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

kfold_mfccs = [a for (a,b) in kfoldset]
kfold_y = [b for (a,b) in kfoldset]

kfold_mfccs = np.array(kfold_mfccs)
kfold_y = to_categorical(np.array(kfold_y))

n_splits_num = 8

# cross validation
skf = KFold(n_splits=n_splits_num, random_state=42, shuffle=True)

num = 0

kfold_loss_list = []
kfold_acc_list = []

for train_index, test_index  in skf.split(kfold_mfccs) :

    num += 1
    
    x_train1 = kfold_mfccs[train_index]
    x_test = kfold_mfccs[test_index]
    y_train1 = kfold_y[train_index]
    y_test = kfold_y[test_index]

    x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1,  train_size=0.8, random_state = 66 ) 

    x_train = np.expand_dims(x_train, -1)

    ip = Input(shape=x_train [0].shape)
        
    m = Conv2D(2, kernel_size=(2,2), activation='relu')(ip)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(4, kernel_size=(2,2), activation='relu')(ip)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(8, kernel_size=(2,2), activation='relu')(ip)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(16, kernel_size=(2,2), activation='relu')(ip)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(32, kernel_size=(2,2), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4,4))(m)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(32*2, kernel_size=(2,2), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4,4))(m)
    m = BatchNormalization(axis=-1)(m)

    m = Conv2D(32*3, kernel_size=(2,2), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4,4))(m)
    m = BatchNormalization(axis=-1)(m)


    m = Flatten()(m)

    m = Dense(64, activation='relu')(m)

    m = Dense(32, activation='relu')(m)

    op = Dense(3, activation='softmax')(m)

    model = Model(ip, op)


    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


    eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다
    reLR = ReduceLROnPlateau(patience=4,verbose=1,factor=0.002) #learning rate scheduler

    history = model.fit(x_train,
                        y_train,
                        epochs=300,
                        batch_size=32,
                        verbose=1,
                        validation_data=(x_val, y_val), callbacks=[eraly_stopping,reLR])



    model.save('../data/project/file/save_model_kfold_new'+str(num)+'.h5')

    loss, acc= model.evaluate(x_test, y_test, batch_size=8)
    # 지표를 만들기 위한 프레딕트 
    print('loss : ',loss)
    print('acc : ',acc)

    kfold_loss_list.append(loss)
    kfold_acc_list.append(acc)

print("로스")
print(kfold_loss_list)
print("ACC")
print(kfold_acc_list)


# 로스
# [3.1941072940826416, 1.3412975072860718, 2.2810473442077637, 2.143760919570923, 2.7560174465179443, 2.6420445442199707, 2.208706855773926, 1.9915393590927124]
# ACC
# [0.7117903828620911, 0.7772925496101379, 0.6681222915649414, 0.7074235677719116, 0.7336244583129883, 0.7030567526817322, 0.719298243522644, 0.780701756477356]

# 8번이 가장 좋음

# 모델을 잘 만들고 K폴드 돌려보자