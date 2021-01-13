
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test,y_test)= mnist.load_data()


# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2])/255.



# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv1D

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=(2), padding='same', input_shape=(28,28)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(100,  kernel_size=(2)) )
model.add(Conv1D(50,  kernel_size=(2)) )
model.add(Conv1D(10,  kernel_size=(2)) )
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=10, batch_size=500, validation_batch_size=0.2, callbacks=earlystopping)

loss, mae = model.evaluate(x_test, y_test, batch_size=500)


print(loss)
print(mae)

y_predict = model.predict(x_test[:10])


print('y_test : ',  y_test[:10].argmax(axis=1))
print('y_predict_argmax : ', y_predict.argmax(axis=1)) 



# LSTM
#0.11151368170976639
# 0.9812999963760376

# conv1d
# 0.12306603789329529
# 0.9605000019073486