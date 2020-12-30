

#   ㅁ
# ㅁ   ㅁ

# 위와 같은 모델 형식

# 앙상블이라고는 볼 수 없다 (앙상블은 위 모델이 엮이는 것이지만 이것은 1개이다)

# 상위 모델이 한개이기 때문에 상위 모델을 엮는 concatenate는 필요없다


import numpy as np

# 1. 데이터

x1 = np.array([ range(100), range(301,401), range(1,101)  ])   # 지금은 (3,100) 이다 
y1 = np.array([range(711,811), range(1,101), range(201,301)])

# x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
# x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, shuffle=False, train_size=0.8) # shuffle False면 섞는다


# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Input
# 텐서플로우 레이어에 여러가지  

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)

# 엮지 않은 모델 1
output1 = Dense(30)(dense1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 엮지 않은 모델 2
output2 = Dense(30)(dense1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=[input1], outputs=[output1,output2])

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train], [y1_train,y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)
# *데이터 2개 이상은 무조건 리스트로 묶는다


# 4. 평가, 예측
loss = model.evaluate([x1_test], [y1_test,y2_test], batch_size=1 )

# 레이어 안 만들고 바로 분기 나누면 모델로 안 잡힌다 

print(loss)

# [2662.756591796875, 1688.963134765625, 973.7931518554688, 1688.963134765625, 973.7931518554688] 
#   1 로스 + 2 로스          1 로스             2 로스         메트릭스 1 로스     메트릭스 2 로스
#    * 대표 로스


print("model.metrics_names : ", model.metrics_names)
# 모델의 메트릭스 네임을 확인할 수 있는 코드


y1_predict, y2_predict = model.predict([x1_test])

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



# 원하는 값 예측 

x_predict3  = np.array([[100,401,101]])


y_test = model.predict([x_predict3])

print(y_test)

