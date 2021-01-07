# 22-3 파일 LSTM 으로 만들기 / Dense와 성능비교 / 다중분류

from sklearn.datasets import load_wine

#1. 데이터 주기
dataset = load_wine()

x = dataset.data
y = dataset.target
# print(x.shape) #(178, 13)
# print(y.shape) #(178,)
# print(x) #전처리가 안 된 것을 확인
# print(y) #순서대로 다중분류되어있으니 셔플을 해야 함

# 나누고
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=66)

# 벡터화하고
from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 범위 0~1사이로
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(113, 13, 1)
print(x_val.shape) #(29, 13, 1)
print(x_test.shape) #(36, 13, 1)

print(y.shape) #(178, 3)
print(y_val.shape) #(178, 3)
print(y_test.shape) #(178, 3)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(120, activation='relu', input_shape=(13,1)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict_argmax: ', y_predict.argmax(axis=1)) 
print('y_test[-5:-1]_argmax: ', y_test[-5:-1].argmax(axis=1)) 

# 22-3 Dense
# loss:  [0.035107001662254333, 0.9722222089767456]
# y_predict_argmax:  [0 2 0 1]
# y_test[-5:-1]_argmax:  [0 2 0 1]

# 33-5 LSTM
# loss:  [0.0969291552901268, 0.9722222089767456]
# y_predict_argmax:  [0 2 0 1]
# y_test[-5:-1]_argmax:  [0 2 0 1]