
# LSTM


import numpy as np


# 1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])


y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_pred = np.array([50,60,70])


# 코딩하시오!!! LSTM
# 나는 80을 원하고 있다

print(x.shape) # (13,3)
print(y.shape) # (13,)

#x = x.reshape(13,3,1) # 3차원 

x = x.reshape(x.shape[0], x.shape[1], 1)

# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()

# 통상적으로 LSTM 2개 이상 사용하면 성능이 떨어짐 하지만 더 좋아질 경우도 있음 이런 것들을 찾아서 하는 것이 나의 할 일 !!!

model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True)) 
# LSTM 레이어를 더 쓰고 싶으면 return_sequences=True를 사용해야 한다 ( 주의 - 마지막 LSTM 레이어는 사용 X )
# LSTM은 3차원을 받아야 하는데 return_sequences=True를 안쓰면 2차원을 내보낸다 
model.add(LSTM(20)) 

model.add(Dense(50, activation='relu'))
# Dense는 차원을 무시하고 모두 받아 쓸 수 있다 

model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


model.summary()

'''


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)


# 4. 평가, 예측

loss = model.evaluate(x,y)
print(loss)

x_pred = x_pred.reshape(1,3,1)
# lstm에 쓸 수 있는 데이터 구조로 변경 (reshape)

result = model.predict(x_pred)
print(result)

# keras23_LSTM3_scale 결과
# 0.10277651250362396
# [[80.14559]]

# keras28_LSTM 결과
# 0.004733389243483543
# [[73.98324]]


'''