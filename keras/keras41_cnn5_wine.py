
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine

datasets = load_wine()

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 66 ) 


print(x.shape)
print(y.shape)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1, 1)
# (x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))

print(x_train.shape)
print(x_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2), padding='same', input_shape=(13,1,1), activation='relu'))
model.add(Conv2D(filters=52, kernel_size=(2), padding='same', input_shape=(13,1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(2), padding='same',  input_shape=(13,1,1), activation='relu'))
model.add(Conv2D(filters=18, kernel_size=(2), padding='same', input_shape=(13,1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])


from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='auto')

model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.3, verbose=2, callbacks=[stop])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=5)

# predict
y_pred = model.predict(x_test)

print("loss : ", loss)
print("accuracy : ", acc)




# CNN - wine
# loss :  0.05633685365319252
# accuracy :  0.9722222089767456


