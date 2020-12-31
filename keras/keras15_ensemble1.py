

# ㅁ   ㅁ
#   ㅁ
# ㅁ   ㅁ

# 위와 같은 모델 형식

import numpy as np

# 1. 데이터

x1 = np.array([ range(100), range(301,401), range(1,101)  ])   # 지금은 (3,100) 이다 
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=False, train_size=0.8) # shuffle False면 섞는다

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, shuffle=False, train_size=0.8)

# 시퀀셜 모델은 앙상블 사용이 어렵다 (모델 2개 합치기 어려움)


# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Input
# 텐서플로우 레이어에 여러가지  

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers from concatenate, Concatenate


# *************

merge1 = concatenate([dense1, dense2])
# 모델 1과 모델 2를 엮는 코드

middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
# 엮은 후 레이어 구성도 가능하다



# 엮은 모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 엮은 모델 분기 2
output2 = Dense(30)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=[input1,input2], outputs=[output1,output2])

model.summary(positions=None)



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)
# *데이터 2개 이상은 무조건 리스트로 묶는다


# 4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test], batch_size=1 )

# 레이어 안 만들고 바로 분기 나누면 모델로 안 잡힌다 

print(loss)

# [2662.756591796875, 1688.963134765625, 973.7931518554688, 1688.963134765625, 973.7931518554688] 
#   1 로스 + 2 로스          1 로스             2 로스         메트릭스 1 로스     메트릭스 2 로스
#    * 대표 로스


print("model.metrics_names : ", model.metrics_names)
# 모델의 메트릭스 네임을 확인할 수 있는 코드


y1_predict, y2_predict = model.predict([x1_test, x2_test])

print(y1_predict)
print("@@@@@@@@@@@@@@@@@@@@")
print(y2_predict)


# 사이킷런
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
    # mean_squared_error는 sklearn에서 mse 만드는 함수 
    # sqrt는 넘파이에 루트 씌우는 함수



rmse1 = RMSE(y1_test,y1_predict)
rmse2 = RMSE(y2_test,y2_predict)
rmse = (rmse1+rmse2) / 2

print("model1_RMSE : ", RMSE(y1_test,y1_predict))
print("model2_RMSE : ", RMSE(y2_test,y2_predict))
print("RMSE : ", rmse)

# print("mse : ", mean_squared_error(y_test, y_predict))

# 아웃풋 모델이 2개인 경우 평균을 내서 구하면 된다


# R2 만드는 법
from sklearn.metrics import r2_score

model1_r2 = r2_score(y1_test , y1_predict)
model2_r2 = r2_score(y2_test , y2_predict)
r2 = (model1_r2 + model2_r2)/2

print("model1_r2 : ", model1_r2)
print("model2_r2 : ", model2_r2)
print("R2 : ", r2)


