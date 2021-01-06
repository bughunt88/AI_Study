# 2개의 모델을 하나는 LSTM, 하나는 Dense로 구현 


import numpy as np
from numpy import array

# 1. 데이터 

x1 = array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[20,30],[30,40],[40,50]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])

y1 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65])
x2_predict = array([65,75,85])


# ****** 중요 **********
# reshape 해주면 x_predict의 값도 reshape 해줘야 한다!!!!


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)


print(x1.shape)
print(x2.shape)


# 2.모델구성

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,LSTM,Input

input1 = Input(shape=(2,1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)

input2 = Input(shape=(3,1))
dense2 = LSTM(10, activation='relu')(input2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)


# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([dense1, dense2])

middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)


# 엮은 모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 엮은 모델 분기 2
output2 = Dense(30)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(1)(output2)

# 모델 선언
model = Model(inputs=[input1,input2], outputs=[output1,output2])

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1,x2], [y1,y2], epochs=100, batch_size=1, validation_split=0.2, verbose=1)


# 4. 평가, 예측
loss = model.evaluate([x1,x2], [y1,y2], batch_size=1 )


print(loss)


x1_predict = x1_predict.reshape(1, 2, 1)
x2_predict = x2_predict.reshape(1, x2_predict.shape[0], 1)


y_predict = model.predict([x1_predict, x2_predict])

print(y_predict)


# y_predict 값 85 나오도록 수정 