import numpy as np

from sklearn.datasets import load_boston

# 1. 데이터

dataset  = load_boston()
# 사이키 런에서 제공하는 데이터 불러오기 
# 데이터 전처리가 되어있는 상태이다 

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)
print("============")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))

print(dataset.feature_names)
# print(dataset.DESCR)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) # shuffle False면 섞는다


# 2. 모델구성

# **************************** 중요
from tensorflow.keras.models import Sequential, Model
# Model은 함수형 모델이다 
from tensorflow.keras.layers import Dense, Input
# Input 레이어가 존재한다


model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 13)) 

# model.add(Dense(128, activation='relu' ,input_shape= (13,))) 

model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))   # 아웃풋 결과가 달라지면 여기도 수정해야한다 



# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2)




# 4. 평가, 예측

loss, mae = model.evaluate(x_test, y_test, batch_size=8)
y_predict = model.predict(x_test)




# *** y_prdict 이랑 y_test 의 shape를 맞춰야 한다 ***


# 사이킷런
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
    # mean_squared_error는 sklearn에서 mse 만드는 함수 
    # sqrt는 넘파이에 루트 씌우는 함수


print(loss)

print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))


# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
