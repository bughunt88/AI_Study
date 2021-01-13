
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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM,Conv1D,MaxPooling1D,Dropout,Flatten

model=Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(13,1))) 
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=52, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
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

print(loss)

# LSTM
# loss:  [0.0969291552901268, 0.9722222089767456]

# conv1d
# [0.00027412563213147223, 1.0]