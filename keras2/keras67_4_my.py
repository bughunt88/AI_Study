# 실습
# 남자 여자 구별
# ImageDataGenerator의 fit 사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten

import PIL
from numpy import asarray
from PIL import Image

import PIL.Image as pilimg


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import string


'''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    vertical_flip= True,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    rotation_range= 5,
    zoom_range= 0.5,
    shear_range= 0.7,
    fill_mode = 'nearest',   
    validation_split=0.2505
 )

test_datagen = ImageDataGenerator(rescale=1./255)   #test data는 따로 튜닝하지 않고 전처리만 해준다.



xy_train = train_datagen.flow_from_directory(
     '../data/image/sex/',        
     target_size = (150,150),
     batch_size= 3000,  
     class_mode='binary', 
     subset = 'training'

 )

xy_test = train_datagen.flow_from_directory(
     '../data/image/sex/',       
     target_size = (150,150),
     batch_size= 3000,
     class_mode='binary', 
     subset = 'validation'
)


print(xy_train[0][0].shape) # (14, 150, 150, 3)
print(xy_train[0][1].shape) # (14,)

np.save('../data/image/sex/numpy/keras67_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/sex/numpy/keras67_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/sex/numpy/keras67_test_x.npy', arr=xy_test[0][0])
np.save('../data/image/sex/numpy/keras67_test_y.npy', arr=xy_test[0][1])
'''


filepath='../data/image/sex/h1.jpeg'
image = pilimg.open(filepath)
image_data = image.resize((150,150))
image_data = np.array(image_data)
image_data = image_data.reshape(1,150,150,3)
answer = [0]
no_answer = [1]




x_train = np.load('../data/image/sex/numpy/keras67_train_x.npy')
y_train = np.load('../data/image/sex/numpy/keras67_train_y.npy')
x_test = np.load('../data/image/sex/numpy/keras67_test_x.npy')
y_test = np.load('../data/image/sex/numpy/keras67_test_y.npy')


# 훈련을 시켜보자! 모델구성 -----------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(128, (7,7), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_acc', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_lacc', factor=0.3, patience=10, mode='max')
filepath = ('../data/modelcheckpoint/k67_-{val_acc:.4f}.hdf5')
mc = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[stop,lr,mc])

loss, acc=model.evaluate(x_test,y_test, batch_size=16)



print("결과")
y_predict = model.predict(image_data)

if y_predict < 0.5:
    print("여자일 확률은 ",np.round((1-y_predict[0][0])*100,2), '%')
else:
    print("남자일 확률은 ",np.round((y_predict[0][0])*100,2), '%')
