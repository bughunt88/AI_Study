# LOSS 함수화 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def custom_mse(y_ture, y_pred):
    return tf.math.reduce_mean(tf.square(y_ture - y_pred))

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

# 2. 모델

model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss=custom_mse, optimizer='adam')

model.fit(x, y, batch_size=1, epochs=1)

loss = model.evaluate(x,y)

print(loss)
