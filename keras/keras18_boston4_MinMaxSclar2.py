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



# 데이터 전처리 (MinMax)
# 통상 적으로 전처리 하면 성능이 좋아진다
# 필수로 해야한다

# x = x/711.
# x = (x-최소) / (최대 - 최소)
# (x - np.min(x)) / (np.max(x) - np.min(x))


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다



# ***** 매우 중요 ******* 
# 전처리의 기준은 트레인!!! 
# 전처리는 트레인만 한다!!!
# 과적합이 생기기 방지하기 위해서 하는 것 

# 사진 3 확인

# 데이터 전처리는 많은 경험과 해봐야 경정 할 수 있다 


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성

# **************************** 중요
from tensorflow.keras.models import Sequential, Model
# Model은 함수형 모델이다 
from tensorflow.keras.layers import Dense, Input
# Input 레이어가 존재한다


model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 13)) 

# model.add(Dense(128, activation='relu' ,input_shape= (13,))) 

model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))   # 아웃풋 결과가 달라지면 여기도 수정해야한다 



# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2)
# 베치 사이즈 작으면 결과는 좋지만 시간이 오래걸린다 



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



print('loss : ', loss)
print('mae : ', mae )


print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))


# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
