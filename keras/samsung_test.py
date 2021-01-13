
import numpy as np


#1 데이터 주고

x = np_load = np.load('../data/npy/x_data.npy')
y = np_load = np.load('../data/npy/y_data.npy')


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=(2), padding='same', input_shape=(5,1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=52, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=18, kernel_size=(2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=20, validation_batch_size=0.2, verbose=1, callbacks=[early_stopping])


model.save('../data/h5/samsung_model.h5')


#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=20)
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




x_predict = np.array([[89800,91200,89100,-1781416,-2190214]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1) # 3차원 

y_predict = model.predict(x_predict)

#[89800	91200	89100 -1781416 -2190214]

print(y_predict)

