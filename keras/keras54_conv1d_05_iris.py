
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
from tensorflow.keras.layers import Dense, Input, LSTM,Conv1D,MaxPooling1D,Dropout,Flatten

model=Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(4,1))) 
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

loss = model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)

# LSTM
# loss:  [0.10291372239589691, 0.9666666388511658]

# conv1d
# loss :  [0.10168784111738205, 0.9666666388511658]