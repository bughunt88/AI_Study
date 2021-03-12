import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston


# 1. 데이터

dataset  = load_boston()

x = dataset.data
y = dataset.target

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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', input_shape=(7,7,1), activation='relu'))
model.summay()