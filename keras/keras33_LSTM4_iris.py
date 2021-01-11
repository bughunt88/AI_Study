#사이킷런 데이터셋 이용 / LSTM 으로 모델링 / Dense와 성능 비교 / 다중분류

import numpy as np
from sklearn.datasets import load_iris 

dataset = load_iris()
x = dataset.data
y = dataset.target
y = np.reshape(y, (150,1))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(96, 4, 1)
print(x_val.shape) #(24, 4, 1)
print(x_test.shape) #(30, 4, 1)

print(y.shape) #(150, 3)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(120, activation='relu', input_shape=(4,1)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict_argmax: ', y_predict.argmax(axis=1)) 
print('y_test[-5:-1]_argmax: ', y_test[-5:-1].argmax(axis=1)) 

# 비교용 22-1-2 Dense
# loss:  [0.11083003133535385, 0.9666666388511658]
# y_predict_argmax:   [2 0 0 2]

# 33-4 LSTM
# loss:  [0.10291372239589691, 0.9666666388511658]
# y_predict_argmax:  [2 0 0 2]


