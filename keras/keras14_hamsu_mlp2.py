# 다:다 mlp 함수형
# keras10_mlp3을 함수형으로 바꾸시오.


# y 행렬의 컬럼의 수를 모델구성의 아웃풋 노드에 넣어줘야 한다 

# 다:다 mlp

import numpy as np

# 1. 데이터 

x = np.array([ range(100), range(301,401), range(1,101)  ])   # 지금은 (3,100) 이다 

y = np.array([range(711,811), range(1,101), range(201,301)])



print(x.shape) # (3,100)
print(y.shape) # (3,100)

# 위에 (3,100)을 (100,3)로 변경해주는 함수
# x = x.reshape((10, 2)) 
x = np.transpose(x)
y = np.transpose(y)


print(x)
print(x.shape) # (100,3)

from sklearn.model_selection import train_test_split
# 싸이킷 런에서 스플릿 해주는 기능이 있다 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=True, random_state = 66 ) 
# random_state 는 랜덤 단수를 고정하는 것이다 매번 돌릴 시 바뀌면 결과값이 달라져서 사용함 

print(x_train.shape) # (80,3)
print(y_train.shape) # (80,3)



# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(3,))
# 인풋 레이어 직접 구성
dense1 = Dense(5, activation='relu')(input1)
# 위에서 지정한 변수 명을 아래에 써줘야 한다
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(3)(dense3)

model = Model(inputs = input1, outputs = outputs)
# 함수형 모델을 지정하려면 인풋과 아웃풋 



# model = Sequential()
# model.add(Dense(10, input_dim= 3)) 
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(3))   # 아웃풋 결과가 달라지면 여기도 수정해야한다 

# 인풋과 아웃풋의 수는 컬럼으로 나뉘어 진다!!!!


# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가, 예측

loss, mae = model.evaluate(x_test,y_test)

print('loss : ', loss)
print('mae : ', mae)


y_predict = model.predict(x_test)

# *** y_prdict 이랑 y_test 의 shape를 맞춰야 한다 ***

print(y_predict)



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
print("mse : ", mean_squared_error(y_predict, y_test))



# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)




