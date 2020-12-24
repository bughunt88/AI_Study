import numpy as np
import tensorflow as tf 

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() # Sequential()는 순차적 모델을 만드는 것이다
model.add(Dense(5, input_dim=1, activation='linear'))# input_dim은 인풋을 알리는 것이다
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #로스는 mse로 계산한다, optimizer는 adam으로 사용한다.
model.fit(x, y, epochs=100, batch_size=1 ) #모델을 실질적으로 훈련시키는 코딩,   epochs - 훈련시키는 횟수를 나타낸다 , batch_size - 데이터를 넣을 떄 넣는 수

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size = 1) 
# 데이터 값으로 loss를 계산한다, evaluate - 평가한다 (지금은 간단하게 사용하지만 나중에는 훈련할 데이터, 결과 값을 나타낼 데이터로 나눈다)
print("loss : ", loss)

result = model.predict([4]) #predict은 입력한 값(예측하고 싶은 값을 넣는다)에 예측 값을 구하는 것이다 
print("result : ", result)