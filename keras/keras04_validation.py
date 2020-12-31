import numpy as np 
import tensorflow as tf

from tensorflow.keras.models import Sequential
# from tensorflow.keras import models 
# from tensorflow import keras
from tensorflow.keras.layers import Dense

# 1. 데이터

# 훈련 데이터
x_train = np.array([1,2,3,4,5]) 
y_train = np.array([1,2,3,4,5])

# 검증 데이터
x_validation = np.array([6,7,8])
y_validation = np.array([6,7,8])

# 테스트 데이터
x_test = np.array([9,10,11]) 
y_test = np.array([9,10,11]) 


# 2. 모델구성

model = Sequential() 
# model = models.Sequential()     - 폴더 구성으로 되어있어서 상위 폴더로 임포트 시키면 하위 폴더 적어줘야 한다!
# model - keras.models.Sequential()

model.add(Dense(5, input_dim=1, activation='relu')) #엑티베이션을 쓰면 적힌 엑티베이션으로 적힌다 안 적으면 디폴트 값으로 계산된다(선형으로 계산됨)
# 통상적으로 relu가 성적이 더 좋다. 

# 레이어를 늘리거나 노드를 늘리면 결과 값이 좋아진다.
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))



# 3. 컴파일, 훈련

# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) 

# metrics는 accuracy, acc 둘다 가능 ***정확도의 값***, 결과 값이 정말 똑같아야 정확도가 올라간다 

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# metrics = mse   로스랑 값 똑같이 나옴

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# metrics = mae   mse에서 파생됨 
# metrics 는 배열로 값을 넣을 수 있다 ex) metrics=['mae','mse']


model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_validation, y_validation))
# validation_data 는 항상 필요하다 머신이 값을 확인하고 예외 값에 대하여 확인하는 과정이다


# **** 통상적으로 val_loss가 더 좋다 loss에 너무 의존하지 말고 loss 와 val_loss를 서로 비교 해야한다


# 평가 예측

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

# result = model.predict([9])

result = model.predict(x_train)


print("result : ", result)



# 제대로 전처리 (validateion_split)
# loss :  10.961244583129883
# mae :  2.407649278640747
# RMSE :  3.3107768583643407
# R2 :  0.8673247641902566


