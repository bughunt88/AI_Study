# hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc

from sklearn.datasets import load_wine

#1. 데이터 주기
dataset = load_wine()

x = dataset.data
y = dataset.target

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

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(120, activation='relu', input_shape=(13,)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=4)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])
print('y_predict_argmax: ', y_predict.argmax(axis=1)) 
print('y_test[-5:-1]_argmax: ', y_test[-5:-1].argmax(axis=1)) 

# 22-3 Dense
# loss:  [0.035107001662254333, 0.9722222089767456]
# y_predict_argmax:  [0 2 0 1]
# y_test[-5:-1]_argmax:  [0 2 0 1]


# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
# plt.plot(x,y) 이걸 넣어도 들어간다

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['loss', 'val loss', 'acc', 'val acc']) #그래프이름
plt.show() 