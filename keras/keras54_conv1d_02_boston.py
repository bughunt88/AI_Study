#사이킷런 데이터셋 이용 / LSTM 으로 모델링 / Dense와 성능 비교

import numpy as np

#1 데이터 주고

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

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
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten



input1 = Input(shape=(13,1))
dense1 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(3,1))(input1)
dense1 = Conv1D(filters=52, kernel_size=2, padding='same', activation='relu', input_shape=(3,1))(input1)
dense1 = Conv1D(filters=30, kernel_size=2, padding='same', activation='relu', input_shape=(3,1))(input1)
dense1 = Conv1D(filters=10, kernel_size=2, padding='same', activation='relu', input_shape=(3,1))(input1)
dense1 = Flatten()(dense1)
dense1 = Dense(24)(dense1)
dense1 = Dense(24)(dense1)
dense1 = Dense(16)(dense1)
dense1 = Dense(16)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=52, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=52)
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

# 33-1 LSTM
# 435
# loss, mae:  20.35696792602539 3.519075393676758
# RMSE:  4.511869802096533
# R2:  0.7180734911817845

# conv1d
# loss, mae:  17.598012924194336 3.2257442474365234
# RMSE:  4.194998609801555
# R2:  0.7562826550970404