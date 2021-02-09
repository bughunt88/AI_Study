# 실습
# cifar_10을 flow로 구성해서 완성
# ImageDataGenerator / fit_generator를 쓸 것

# cifar10를 flow로 구성해서 완성
# male, female >> Imagegenerator, fit_generator 적용해서 완성

from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=42)
print(x_train.shape, y_train.shape)    
print(x_test.shape, y_test.shape)       
print(x_valid.shape, y_valid.shape)    

print(len(x_train)) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_valid = to_categorical(y_valid)

print(y_train.shape, y_test.shape, y_valid.shape)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.0,
    shear_range=0.7,
    fill_mode='nearest'
)
etc_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=16)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=16)
valid_generator = etc_datagen.flow(x_valid, y_valid)

model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(10, activation='softmax'))

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
hist = model.fit_generator(train_generator, epochs=50, \
    steps_per_epoch = len(x_train) // 16 , validation_data=valid_generator, callbacks=[es, lr])

loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)    

# cifa10 
# loss :  1.0630311965942383
# acc :  0.661899983882904