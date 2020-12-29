# 모델 구성 (fit)에서 verbose(0~3)을 넣으면 콘솔에 나타나는 데이터가 다르다 

import numpy as np

# 1. 데이터 

x = np.array([ range(100), range(301,401), range(1,101) , range(201,301), range(101,201) ])   # 지금은 (2,10) 이다 

y = np.array([range(711,811), range(1,101)])

print(x.shape) # (5,100)
print(y.shape) # (2,100)


x_pred2 = np.array([100,402,101,100,401])

print(x_pred2.shape)

x_pred2 = x_pred2.reshape(1,5)


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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim= 5)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))   # 아웃풋 결과가 달라지면 여기도 수정해야한다 

# 인풋과 아웃풋의 수는 컬럼으로 나뉘어 진다!!!!


# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0)


# verbose=0 을 넣으면 훈련하는 과정을 안보인다
# 0으로 넣으면 과정이 안보이는 대신 속도는 빨리진다
 


# verbose = 0 : 과정이 보이지 않는다

# verbose = 1 :
# Epoch 100/100
# 64/64 [==============================] - 0s 1ms/step - loss: 1.7370e-09 - mae: 3.3557e-05 - val_loss: 1.6258e-09 - val_mae: 3.0488e-05

# verbose = 2 :
# Epoch 100/100
# 64/64 - 0s - loss: 1.3620e-09 - mae: 2.1065e-05 - val_loss: 2.1835e-09 - val_mae: 2.5600e-05

# verbose = 3 : 
# Epoch 100/100

# verbose 의 디폴트는 1



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



print('loss : ', loss)
print('mae : ', mae )


print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))



# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# 원하는 값 예측 

x_predict  = np.array([[100,401,101,301,201],  [105,406,106,306,206]])

y_predict = model.predict(x_predict)


print("x_predic : ", y_predict)

y_pred2 = model.predict(x_pred2)

print("y_pred2 : ", y_pred2)

# 예상 값 [[811, 101],[816,106]]



# loss :  1.4178589413660347e-09
# mae :  2.4537741410313174e-05
# RMSE :  3.7654468337435804e-05
# mse :  1.4178589857749556e-09
# R2 :  0.999999999998206
# x_predic :  [[811.00006  100.999985],[816.00006  105.99998 ]]