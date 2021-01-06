# 2개의 모델을 하나는 LSTM, 하나는 Dense로 구현 


import numpy as np
from numpy import array

# 1. 데이터 

x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])


# ****** 중요 **********
# reshape 해주면 x_predict의 값도 reshape 해줘야 한다!!!!


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)

print(x1.shape)
print(x2.shape)


# 2.모델구성

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,LSTM,Input

input1 = Input(shape=(3,1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)


# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([dense1,dense2])

middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(1)(middle1)

# 모델 선언
model = Model(inputs=[input1,input2], outputs=middle1)

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1,x2], y, epochs=100, batch_size=1, validation_split=0.2, verbose=1)


# 4. 평가, 예측
loss = model.evaluate([x1,x2], y, batch_size=1 )

print(loss)

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3)


y_predict = model.predict([x1_predict, x2_predict])

print(y_predict)


# y_predict 값 85 나오도록 수정 


# LSTM 왼쪽  
# [805.1168823242188, 13.503647804260254]
# [[9.503746]]


# LSTM 오른쪽
# [816.5464477539062, 13.672287940979004]
# [[9.030447]]