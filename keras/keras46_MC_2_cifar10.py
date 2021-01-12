from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test,y_test)= cifar10.load_data()


# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], x_train.shape[3])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], x_train.shape[3])/255.

# x 같은 경우 색상의 값이기 때문에 255가 최고 값


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(filters=52, kernel_size=(2), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(2), padding='same',  input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(filters=18, kernel_size=(2),padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(monitor='val_loss', patience=16, mode='auto')

modelpath = '../data/modelcheckpoint/k46_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=60, batch_size=400, validation_split=0.2, verbose=1, callbacks=[es,cp])


loss, mae = model.evaluate(x_test, y_test, batch_size=400)

print(loss)
print(mae)

y_predict = model.predict(x_test[:10])



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