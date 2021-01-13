import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test,y_test)= fashion_mnist.load_data()

# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2])/255.

# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=(2), padding='same', input_shape=(28,28), activation='relu'))
model.add(Conv1D(filters=52, kernel_size=(2), padding='same', input_shape=(28,28), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=(2), padding='same',  input_shape=(28,28), activation='relu'))
model.add(Conv1D(filters=18, kernel_size=(2),padding='same', input_shape=(28,28), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])


from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='auto')

model.fit(x_train, y_train, epochs=60, batch_size=10, validation_split=0.3, verbose=1, callbacks=[stop])


loss, mae = model.evaluate(x_test, y_test, batch_size=10)

print(loss)
print(mae)

y_predict = model.predict(x_test[:10])


print('y_test : ',  y_test[:10].argmax(axis=1))
print('y_predict_argmax : ', y_predict.argmax(axis=1)) 


# fashion - lstm
# 0.33231616020202637
# 0.885699987411499
# y_test :  [9 2 1 1 6 1 4 6 5 7]
# y_predict_argmax :  [9 2 1 1 6 1 4 6 5 7]

# conv1d
# 0.3443305790424347
# 0.878000020980835
# y_test :  [9 2 1 1 6 1 4 6 5 7]
# y_predict_argmax :  [9 2 1 1 6 1 4 6 5 7]