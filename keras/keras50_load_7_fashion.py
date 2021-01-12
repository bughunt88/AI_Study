
import numpy as np
import matplotlib.pyplot as plt


x_train = np.load('../data/npy/fashion_x_train.npy')
y_train = np.load('../data/npy/fashion_y_train.npy')

x_test = np.load('../data/npy/fashion_x_test.npy')
y_test = np.load('../data/npy/fashion_y_test.npy')



# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)/255.

# x 같은 경우 색상의 값이기 때문에 255가 최고 값


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(filters=52, kernel_size=(2), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(2), padding='same',  input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(filters=18, kernel_size=(2),padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
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

# fashion - cnn
# 0.24186135828495026
# 0.9276999831199646
# y_test :  [9 2 1 1 6 1 4 6 5 7]
# y_predict_argmax :  [9 2 1 1 6 1 4 6 5 7]