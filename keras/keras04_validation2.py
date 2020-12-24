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
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
# validation_split 은 주어진 데이터에서 자동으로 쪼개서 쓴다

# 4. 평가, 예측
result = model.evaluate(x_test,y_test, batch_size=1)
#위에 컴파일에서 넣은 값들이 자동으로 들어간다 ex) loss, metrics

print("results : ", result)

y_pred = model.predict(x_pred)

print("y_pred : ", y_pred)




