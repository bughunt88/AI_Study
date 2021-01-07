# hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc



#1. 데이터
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터 주고
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

#전처리(y벡터화, 트레인테스트나누기, 민맥스스케일러)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=33)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=33)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#모델 구성
input1 = Input(shape=(30,))
dense1 = Dense(120, activation='relu')(input1)
dense1 = Dense(120)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
output1 = Dense(2, activation='sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=5, mode='min')
hist =model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#검증, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])

# print('y_predict: ', y_predict)
# print('y_predict_argmax: ', y_predict.argmax(axis=1)) #0이 열, 1이 행
# print('y_test[-5:-1]: ',y_test[-5:-1])


# print(y_predict.argmax(axis=1).shape) #(4,)
# print(y_test[-5:-1].shape) #(4,2)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_acc'])
# plt.plot(x,y) 이걸 넣어도 들어간다

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['loss', 'acc', 'val_loss', 'val acc']) #그래프이름
plt.show() 