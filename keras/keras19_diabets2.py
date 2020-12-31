# 실습 : 19_1번 부터 EarlyStopping 까지 
# 총 6개 파일 완성하시오.


import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)
print(np.max(x), np.min(y))

print(dataset.feature_names)
# 컬럼 이름 불러오는 코드
print(dataset.DESCR)
# 컬럼에 대한 설명 (주석)



x = x/0.198787989657293


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) # shuffle False면 섞는다


# 2. 모델구성

# **************************** 중요
from tensorflow.keras.models import Sequential, Model
# Model은 함수형 모델이다 
from tensorflow.keras.layers import Dense, Input
# Input 레이어가 존재한다


model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 10)) 

# model.add(Dense(128, activation='relu' ,input_shape= (13,))) 

model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))

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


print("loss : ",loss)

print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))


# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


#2950.79541015625
#RMSE :  54.321224951269016
#R2 :  0.4622316404403659