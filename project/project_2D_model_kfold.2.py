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

import warnings

warnings.filterwarnings('ignore')

trainset = np.load('../data/project/data/train_data.npy',allow_pickle=True)
testset = np.load('../data/project/data/test_data.npy',allow_pickle=True)

kfoldset = np.load('../data/project/data/kfold_data.npy',allow_pickle=True)


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


    print(x_train1.shape)

    print("########################")
    print(x_train.shape)
    print(y_train.shape)

    print(x_val.shape)

    print("#####################@@@@@@@@@@@@")

    print(x_train.shape)
    print(y_train.shape)

    '''
    train_mfccs: (1657, 120, 650)     
    train_y: (1657, 3)
    test_mfccs: (173, 120, 650)       
    test_y: (173, 3)
    train X shape: (1657, 120, 650, 1)
    test X shape: (173, 120, 650, 1)  
    (120, 650, 1)
    '''
    
    x_train = np.expand_dims(x_train, -1)

    print(x_train.shape)
    print(y_train.shape)
    print("########################")
    print(x_train[0].shape)

    ip = Input(shape=x_train[0].shape)

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


    eraly_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') # mode는 min,max,auto 있다
    reLR = ReduceLROnPlateau(patience=5,verbose=1,factor=0.5) #learning rate scheduler

    history = model.fit(x_train,
                        y_train,
                        epochs=500,
                        batch_size=8,
                        verbose=1,
                        validation_data=(x_val, y_val), callbacks=[eraly_stopping,reLR])


    model.save('../data/project/file/save_model_kfold'+str(num)+'.h5')

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


'''
    DATA_DIR = '../data/project/predict/'

    for filename in os.listdir(DATA_DIR):
        filename = normalize('NFC', filename)

        wav, sr = librosa.load(DATA_DIR + filename)
        mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=120, n_fft=1000, hop_length=120)

        #S_1 = librosa.power_to_db(mfcc, ref=np.max)
        #mfcc = _normalize(S_1)

        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        padded_mfcc = pad2d(mfcc, 650)
        padded_mfcc= np.expand_dims(padded_mfcc, 0)

        #librosa.display.specshow(padded_mfcc, sr=16000, x_axis='time')

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


# 로스
# [4.083415985107422, 1.4742237329483032, 2.5895190238952637, 1.9262902736663818, 2.1092889308929443, 2.734142780303955, 3.2057576179504395, 2.1240248680114746]
# ACC
# [0.751091718673706, 0.7598253488540649, 0.6331877708435059, 0.7074235677719116, 0.72052401304245, 0.6681222915649414, 0.7061403393745422, 0.6666666865348816]

# 2번 모델이 가장 좋음