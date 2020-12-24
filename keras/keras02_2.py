import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential

# from tensorflow.keras import models 
# from tensorflow import keras

from tensorflow.keras.layers import Dense

#1. 데이터

# 훈련 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([2,4,6,8,10,12,14,16,18,20])


#테스트 데이터
x_test = np.array([101,102,103,104,105,106,107,108,109,110]) 
y_test = np.array([111,112,113,114,115,116,117,118,119,120]) 

x_predict = np.array([111,112,113])

#2. 모델구성
model = Sequential() 
# model = models.Sequential()     - 폴더 구성으로 되어있어서 상위 폴더로 임포트 시키면 하위 폴더 적어줘야 한다!
# model - keras.models.Sequential()

model.add(Dense(5, input_dim=1, activation='relu')) #엑티베이션을 쓰면 적힌 엑티베이션으로 적힌다 안 적으면 디폴트 값으로 계산된다(선형으로 계산됨)
#통상적으로 relu가 성적이 더 좋다. 

#레이어를 늘리거나 노드를 늘리면 결과 값이 좋아진다.

model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))


model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

# 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result = model.predict(x_predict)
print("result : ", result)

