from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test,y_test)= cifar100.load_data()


# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], x_train.shape[3])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], x_train.shape[3])/255.

# x 같은 경우 색상의 값이기 때문에 255가 최고 값


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM


model = Sequential()

model.add(Conv2D(filters=200, kernel_size=(2), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(filters=100, kernel_size=(2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(2), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(filters=52, kernel_size=(2), padding='same',  activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(100, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])


from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='auto')

model.fit(x_train, y_train, epochs=50, batch_size=200, validation_split=0.2, verbose=1, callbacks=[stop])


loss, mae = model.evaluate(x_test, y_test, batch_size=200)

print(loss)
print(mae)

y_predict = model.predict(x_test[:10])


print('y_test : ',  y_test[:10].argmax(axis=1))
print('y_predict_argmax : ', y_predict.argmax(axis=1)) 

# cifar100 - cnn
# 2.353292465209961
# 0.4293999969959259
# y_test :  [49 33 72 51 71 92 15 14 23  0]
# y_predict_argmax :  [49 65 72 91 71 79 80 78 71 83]