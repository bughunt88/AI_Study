# 인공지능계의 hello world라 불리는 mnist
# # 실습!! 완성하시오!!
# 지표는 acc

# 응용
# y_test 10개와 y_test 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,,,,)
# y_pred[:10] = (?,?,?,?,?,?,?,,,,) 

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

import tensorflow as tf
# 그래픽 카드 다중 일 때 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)



(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_pred = x_test[:10]

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8, shuffle = True)

# print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])
# print(x_train[0].shape) # (28,28)

# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0]) # gray 안넣어주면 컬러로 나오는데 제대로 된건 아님
# plt.show()

# 민맥스 스케일을 못 쓰므로 /255. 해준다
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],x_pred.shape[2],1)/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)/255.
# (x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
    tf.distribute.HierarchicalCopyAllReduce()
)

# Then, log the results on a shared location, write TensorBoard logs, etc
with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters = 80, kernel_size = (2,2), padding = 'same', strides = 1, input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,2))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    model.summary()

    #3. 컴파일 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train,y_train, epochs = 1000, batch_size = 32 ,validation_data=(x_val,y_val), verbose = 2, callbacks = [es])

#4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size = 32)
print('loss : ', loss[0])
print('acc : ', loss[1])
y_pred = model.predict(x_pred)