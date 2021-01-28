# 아래 모델을 완성하시오 (지표는 acc >= 0.985)
# checkpoint에 날짜와 시간을 표시한다.

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)  > 0 ~ 9 다중 분류

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 5
# print(x_train[0].shape)               # (28, 28)

# plt.imshow(x_train[0], 'gray')  # 0 : black, ~255 : white (가로 세로 색깔)
# plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
# plt.show()  

# preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
# 4차원 만들어준다. float타입으로 바꾸겠다. -> /255. -> 0 ~ 1 사이로 수렴됨
x_test = x_test.reshape(10000, 28, 28, 1)/255. 
# x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],1)

# y >> OnHotEncoding

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
# print(y_train[0])       # [5]
# print(y_train.shape)    # (60000, 1)
# print(y_test[0])        # [7]
# print(y_test.shape)     # (10000, 1)

encoder = OneHotEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()    #toarray() : list 를 array로 바꿔준다.
# print(y_train)
# print(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)


#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=16, kernel_size=(4,4), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.1))
model.add(Flatten())

model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

# model.summary()


# ModelCheckPoint

import datetime 

# date_now = datetime.datetime.now()
# print(date_now)     # 컴퓨터에 지정된 현재 시간이 출력 됨 : 2021-01-27 10:13:00.489878
# nowDatetime = date_now.strftime('%m%d_%H%M%S')
# print(nowDatetime)  
# date_time = date_now.strftime("%m%d_%H%M")
# print(date_time)    # 0127_1013

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath='../data/modelcheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k45_", '{timer}', filename])

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils
@keras_export('keras.callbacks.ModelCheckpoint')
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, timer=datetime.datetime.now().strftime('%m%d_%H%M%S'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath
        
cp = MyModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=5, mode='max')

# Compile, Train            

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es, cp])

# Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("accuracy : ", result[1])