# Sequential 모델이 아닌 함수형 모델 만드는 법 


import numpy as np

# 1. 데이터 

x = np.array([ range(100), range(301,401), range(1,101) , range(201,301), range(101,201) ])   # 지금은 (2,10) 이다 

y = np.array([range(711,811), range(1,101)])

print(x.shape) # (5,100)
print(y.shape) # (2,100)


# 위에 (2,10)을 (10,2)로 변경해주는 함수
# x = x.reshape((10, 2)) 
x = np.transpose(x)
y = np.transpose(y)

print(x.shape) # (100.5)
print(y.shape) # (100.2)

from sklearn.model_selection import train_test_split
# 싸이킷 런에서 스플릿 해주는 기능이 있다 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=True, random_state = 66 ) 
# random_state 는 랜덤 단수를 고정하는 것이다 매번 돌릴 시 바뀌면 결과값이 달라져서 사용함 

print(x_train.shape) # (80,5)
print(y_train.shape) # (80,2)


# 2. 모델구성

# ****************************

from tensorflow.keras.models import Sequential, Model
# Model은 함수형 모델이다 
from tensorflow.keras.layers import Dense, Input
# Input 레이어가 존재한다

input1 = Input(shape=(5,))
# 인풋 레이어 직접 구성
dense1 = Dense(5, activation='relu')(input1)
# 위에서 지정한 변수 명을 아래에 써줘야 한다
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)

model = Model(inputs = input1, outputs = outputs)
# 함수형 모델을 지정하려면 인풋과 아웃풋 

model.summary()
# 무슨 모델인지 알고 싶으면  .summary() 을 사용


# 위에 모델 구성과, 아래 모델 구성이 같다. 대신 모델 방식이 다르

'''
model = Sequential()
#model.add(Dense(10, input_dim= 5)) 
model.add(Dense(5, activation='relu' ,input_shape= (5,))) 
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))  

model.summary()
'''

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0)

# 4. 평가, 예측

loss, mae = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# *** y_prdict 이랑 y_test 의 shape를 맞춰야 한다 ***



# 사이킷런
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
    # mean_squared_error는 sklearn에서 mse 만드는 함수 
    # sqrt는 넘파이에 루트 씌우는 함수


print(y_test)
print(y_predict)


print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))



# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
