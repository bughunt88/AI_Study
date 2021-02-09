# 실습
# cifar_10을 flow로 구성해서 완성
# ImageDataGenerator / fit_generator를 쓸 것



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import cifar10


(x_train,y_train),(x_valid,y_valid) = cifar10.load_data()

print(x_train.shape)    #50000,32,32,3
print(y_train.shape)    #50000,1
print(x_valid.shape)     #10000,32,32,3

idg = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기 
    vertical_flip=True, # 수직 뒤집기 
    width_shift_range=0.1, # 수평 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=5, # 회전 
    zoom_range=1.2, # 확대
    shear_range=0.7, # 층 밀리기 강도?
    fill_mode='nearest' # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
)

idg2 = ImageDataGenerator()

train_generator = idg.flow(x_train,y_train,batch_size=32, seed = 2048)
valid_generator = idg2.flow(x_valid,y_valid)

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size =(3,3), activation='relu', padding = 'same', 
                                        input_shape=(32,32,3)))
model.add(BatchNormalization())                                  
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

                            
model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 160)
lr = ReduceLROnPlateau(patience= 90, factor=0.5)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),
                metrics=['acc'])
history = model.fit_generator(train_generator,epochs=2000, callbacks=[early_stopping,lr])


loss, acc = model.evaluate(valid_generator)
print("loss : ", loss)
print("acc : ", acc)    

# loss :  1.1506491899490356
# acc :  0.6115999817848206