
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

model.add(Dense(1))

model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mae'])


from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='auto')

model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.2, verbose=2, callbacks=[stop])


#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=5)

y_pred = model.predict(x_test)



# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
print('RMSE: ',RMSE(y_test,y_pred))

# R2
from sklearn.metrics import r2_score
def R2(y_test,y_pred):
    return r2_score(y_test,y_pred)
print('R2: ', R2(y_test,y_pred))


# CNN - boston
# RMSE:  3.048824990483019
# R2:  0.8874889863150697



