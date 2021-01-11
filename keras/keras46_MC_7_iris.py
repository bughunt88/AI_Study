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


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
modelpath = './ModelCheckPoint/k46_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp])

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)


# 시각화
import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()

#plt.title('손실비용')
plt.title('Cost Loss')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'],marker='.',c='red',label='acc')
plt.plot(hist.history['val_acc'],marker='.',c='blue',label='acc')
plt.grid()

# plt.title('정확도')
plt.title('Accuracy')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()