
# LSTM -> conv1d


import numpy as np


# 1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])


y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_pred = np.array([50,60,70])


x = x.reshape(13,3,1) # 3차원 


# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(3,1))) 
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.2))
model.add(Conv1D(filters=52, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.2))
model.add(Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)


# 4. 평가, 예측

loss = model.evaluate(x,y)
print(loss)

# keras23_LSTM3_scale 결과
# 0.10277651250362396
# [[80.14559]]