# 실습
# 남자 여자 구별
# ImageDataGenerator의 fit 사용해서 완성
# vgg16으로 만들어봥!!


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D

from keras.utils import np_utils

import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet import ResNet101,preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

x_train = np.load('../data/image/sex/numpy/keras67_train_x.npy')
y_train = np.load('../data/image/sex/numpy/keras67_train_y.npy')
x_test = np.load('../data/image/sex/numpy/keras67_test_x.npy')
y_test = np.load('../data/image/sex/numpy/keras67_test_y.npy')

print(x_train.shape)


x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)


vgg16 = VGG16(weights='imagenet',input_shape =(255,255,3),include_top=False)
vgg16.trainalbe = False
model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation= 'sigmoid'))


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor= 'val_loss', patience=120)
lr = ReduceLROnPlateau(monitor='val_loss', patience=60, factor=0.5)
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['acc'])
history = model.fit(x_train,y_train, epochs=500, validation_data=(x_test,y_test),
callbacks=[es,lr])

loss, acc=model.evaluate(x_test,y_test, batch_size=16)

print("y 값")
print(y_test[0])
print("y 계산 값")
print(model.predict(x_test[0], verbose=True))

y_predict = model.predict(x_test[0], verbose=True)

print(np.where(y_predict> 0.5, 1, 0))

# male_female(fit_generator)
# loss :  1.7091652154922485
# acc :  0.7373272180557251

# y 값
# [1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 0. 1.]

# y 계산 값
# [[1][1][1][1][0][0][0][1][1][1][1][1][0][1]]
# male_female(fit)
# loss :  2.4603981971740723
# acc :  0.5898617506027222
