# 실습
# R2를 음수가 아닌 0.5 이하로 줄이기
# 1. 레이어는 인풋과 아웃풋을 포함 6개 이상
# 2. batch_size = 1
# 3. epochs = 100 이상
# 4. 데이터 조작 금지
# 5. 히든레이어의 노드의 갯수는 10 이상

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array
# np.array()
# array()


# 1. 데이터

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])

x_pred = array([16,17,18])

# 2. 모델 구성

model = Sequential()
model.add(Dense(10, input_dim = 1, activation="relu"))

model.add(Dense(10000000))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))


model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=113, batch_size=1, validation_split=0.2)
# validation_split 은 주어진 데이터에서 자동으로 쪼개서 쓴다

# 4. 평가, 예측
result = model.evaluate(x_test,y_test, batch_size=1)
#위에 컴파일에서 넣은 값들이 자동으로 들어간다 ex) loss, metrics
print("mse, mae : ", result)


# 결과 값 뽑기
y_predict = model.predict(x_test)
# print("y_pred : ", y_predict)


# 사이킷런
from sklearn.metrics import mean_squared_error

# def는 함수 만드는 것
# rmse라는 함수를 만들어서 mse를 만들고 루트를 씌우는 과정을 하는 것 임

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
    # mean_squared_error는 sklearn에서 mse 만드는 함수 
    # sqrt는 넘파이에 루트 씌우는 함수

print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))



# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)