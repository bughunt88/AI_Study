#텐서플로우 데이터셋 이용 / LSTM 으로 모델링 / Dense와 성능 비교 / 회귀

import numpy as np

#1 데이터 주고

from tensorflow.keras.datasets import boston_housing
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

print(x_train.shape) 
print(x_test.shape) 

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(13,1))
dense1 = LSTM(13, activation='relu')(input1)
dense1 = Dense(13, activation='relu')(dense1)
dense1 = Dense(26, activation='relu')(dense1)
dense1 = Dense(26, activation='relu')(dense1)
dense1 = Dense(13, activation='relu')(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=2000, batch_size=26, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=26)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

# 비교용 20-1 Dense
# loss, mae:  11.377790451049805 2.260841131210327
# RMSE:  3.3730985125849697
# R2:  0.8633197073667653

# 33-1-1 LSTM
# 435
# loss, mae:  17.13974952697754 3.1163034439086914
# RMSE:  4.1400179244603565
# R2:  0.7626292300068274

# 33-1-2 LSTM
# 240
# loss, mae:  33.55441665649414 4.119900703430176
# RMSE:  5.792617422256556
# R2:  0.5969141600577362