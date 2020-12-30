
# ***매우 중요하다!!!
# 모델이 이상이 있다면 서머리를 보고 판단해야한다!


import numpy as np
import tensorflow as tf 

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() # Sequential()는 순차적 모델을 만드는 것이다
model.add(Dense(7, input_dim=1, activation='linear'))# input_dim은 인풋을 알리는 것이다
model.add(Dense(8, activation='linear'))
model.add(Dense(4, name='aaa'))
model.add(Dense(1))


model.summary()



# 실습2 + 과제
# ensemble 1,2,3,4 에 대해 서머리를 계산하고
# 이해한 것을 과제로 제출할 것

# layer를 만들 때 'name' 이런놈에 대해 확인하고 설명할 것
# 레이어의 name이라는 파라미터를 왜 쓰는지, name을 반드시 써야할 때가 있다. 그때를 말하여라.

