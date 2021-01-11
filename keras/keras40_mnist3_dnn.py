# 주말과제
# dance 모델로 구성 input_shape=(764,)

# 주말과제
# dance 모델로 구성 input_shape=(764,)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test,y_test)= mnist.load_data()

# plt.imshow(x_train[1])
# plt.show()



# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.
# (x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))


print(x_train.shape)
print(y_train.shape)


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical # OneHotEncoding from tensorflow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




from tensorflow.keras.models import Sequential,  Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM, Input



inputs = Input(shape=(784,))

dense1 = Dense(128, activation='relu')(inputs)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(48, activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dropout(0.2)(dense1)

outputs = Dense(10,activation='softmax')(dense1)

model = Model(inputs= inputs, outputs = outputs)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=80, validation_split=0.2, verbose=1, callbacks=[stop])


#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=80)
y_predict = model.predict(x_test[:10])

print(loss)
print(mae)

print('y_test : ',  y_test[:10].argmax(axis=1))
print('y_predict_argmax : ', y_predict.argmax(axis=1)) 

# CNN
# 0.02563497982919216
# 0.991599977016449
# y_test :  [7 2 1 0 4 1 4 9 5 9]
# y_predict_argmax :  [7 2 1 0 4 1 4 9 5 9]


# DNN
# 0.13371659815311432
# 0.9804999828338623
# y_test :  [7 2 1 0 4 1 4 9 5 9]
# y_predict_argmax :  [7 2 1 0 4 1 4 9 5 9]