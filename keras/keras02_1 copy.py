import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 훈련 데이터
x_train = np.array([[1,2,3,4,5], [6,7,8,9,10]]) 
y_train = np.array([[6,7,8,9,10], [11,12,13,14,15]])

x_train = tf.expand_dims(x_train, axis=-1)
y_train = tf.expand_dims(y_train, axis=-1)

print(x_train.shape)
print(y_train.shape)

#  x = 2970,30,1
#  y = 2970,30,1


# (2, 5, 1)
# (2, 5, 1)


# 결과 데이터 

x_predict = np.array([16,17,18,19,20]) 

x_predict = tf.expand_dims(x_predict, axis=-1)

print(x_predict.shape)


model = Sequential() 
model.add(Dense(5, input_shape=[5, 1], activation='relu')) 
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

result = model.predict(x_predict)
print("result : ", result)

 #[[12.225188],[12.943766],[13.662343],[14.380919],[15.099498]]

