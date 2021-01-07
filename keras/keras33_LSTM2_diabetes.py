#사이킷런 데이터셋 이용 / LSTM 으로 모델링 / Dense와 성능 비교 / 회귀모델

import numpy as np
from sklearn.datasets import load_diabetes #당뇨병 수준

#데이터 불러오고 , 이건 사이킷런의 데이터 불러오는 방법
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(442, 10) (442,)

#트레인 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle=True, random_state=66)

#데이터 전처리 (MinMaxScaler를 이용해서 , 기준은 x_train으로)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# print(x_train.shape) #(282, 10, 1)
# print(x_val.shape) #(71, 10, 1)
# print(x_test.shape) #(89, 10, 1)

#모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(80, input_shape=(10,1), activation = 'relu'))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

#컴파일, 훈련 (Earlystopping 적용)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=16, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=60, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=60)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)


# 비교용 19-6 Dense
# loss, mae:  3121.9990234375 46.306800842285156
# RMSE:  55.87484721026777
# R2:  0.5189554519135346

# 33-2 LSTM
# loss, mae:  4015.6279296875 52.748226165771484
# RMSE:  63.36898127909322
# R2:  0.3812630520749578