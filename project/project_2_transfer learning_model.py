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
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D, Activation, Add, GlobalAveragePooling2D
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


print(train_X_ex[0].shape)

x_train, x_val, y_train, y_val = train_test_split(train_X_ex, train_y,  train_size=0.8, random_state = 66 ) 


# 모델
input_tensor = Input(shape=train_X_ex[0].shape, dtype='float32', name='input')

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   

def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x

def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x
 
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)
 
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(3, activation='softmax')(x)

resnet50 = Model(input_tensor, output_tensor)

resnet50.summary()

model = resnet50

'''
ip = Input(shape=train_X_ex[0].shape)

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
'''

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


eraly_stopping = EarlyStopping(monitor='loss', patience=120, mode='auto') # mode는 min,max,auto 있다
reLR = ReduceLROnPlateau(patience=30,verbose=1,factor=0.8) #learning rate scheduler

history = model.fit(x_train,
                    y_train,
                    epochs=5000,
                    batch_size=32,
                    verbose=1,
                    validation_data=(x_val, y_val), callbacks=[eraly_stopping,reLR])


model.save('../data/project/file/save_model.h5')

loss, acc= model.evaluate(test_X_ex, test_y, batch_size=8)
# 지표를 만들기 위한 프레딕트 
print('loss : ',loss)
print('acc : ',acc)


min_level_db = -100
 
def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


print(x_train.shape)
print(y_train.shape)


DATA_DIR = '../data/project/predict/'

for filename in os.listdir(DATA_DIR):
    filename = normalize('NFC', filename)

    wav, sr = librosa.load(DATA_DIR + filename)
    mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=120, n_fft=1000, hop_length=120)
    #mfcc = librosa.feature.mfcc(wav, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)

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
loss :  1.0676013231277466
acc :  0.7919074892997742
'''