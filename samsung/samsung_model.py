import io
import pandas as pd
import numpy as np

#from google.colab import drive
#drive.mount('/content/drive')

# 데이터 변수
size = 5 #30


#total_data = np.load('/content/drive/My Drive/samsung_data.npy')
#total_kodex_data = np.load('/content/drive/My Drive/kodex_data.npy')

total_data = np.load('./samsung_data.npy')
total_kodex_data = np.load('./kodex_data.npy')


y_data = total_data


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)
total_data = split_x(total_data,size)
total_kodex_data  = split_x(total_kodex_data,size)

# 삼성 데이터
x = total_data[:-2,:size, :]
y = []
for n in range(2, len(y_data) - size + 1):
  y.append([y_data[n+size-2][0],y_data[n+size-1][0]])
y = np.array(y)
x_pred = total_data[-1,:size,:]

print(x_pred)
print(x[-1])
print(y[-1])

x1_shape = x.shape[1]
x2_shape = x.shape[2]

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred = x_pred.reshape(1,x_pred.shape[0]* x_pred.shape[1])


# 코덱스 데이터 
kodex_x = total_kodex_data[:-2,:size, :]
kodex_x_pred = total_kodex_data[-1,:size,:]

kodex1_shape = kodex_x.shape[1]
kodex2_shape = kodex_x.shape[2]

#print(kodex_x_pred)
print(kodex_x[-1])
print(kodex_x_pred)

kodex_x = kodex_x.reshape(kodex_x.shape[0], kodex_x.shape[1]*kodex_x.shape[2])
kodex_x_pred = kodex_x_pred.reshape(1,kodex_x_pred.shape[0]* kodex_x_pred.shape[1])


# 데이터 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, kodex_train, kodex_test = train_test_split(x, y, kodex_x, train_size=0.7, shuffle=False)




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x1_shape, x2_shape)
x_test = x_test.reshape(x_test.shape[0], x1_shape, x2_shape)


kodex_scaler = MinMaxScaler()
kodex_scaler.fit(kodex_train)
kodex_train = kodex_scaler.transform(kodex_train)
kodex_test = kodex_scaler.transform(kodex_test)
kodex_x_pred = kodex_scaler.transform(kodex_x_pred)

kodex_train = kodex_train.reshape(kodex_train.shape[0], kodex1_shape, kodex2_shape)
kodex_test = kodex_test.reshape(kodex_test.shape[0], kodex1_shape, kodex2_shape)

# 모델
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,Input,Activation, LSTM


#model = load_model('/content/drive/MyDrive/check_point_best.h5')
model = load_model('./check_point_best.h5')


model.compile(loss='mse', optimizer='adam', metrics=['mae'])


#평가, 예측
loss, mae = model.evaluate([x_test,kodex_test], y_test, batch_size=4)
print('loss, mae: ', loss, mae)

y_predict = model.predict([x_test,kodex_test])

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)


x_predict = x_pred.reshape(1,x1_shape,x2_shape)

kodex_x_predict = kodex_x_pred.reshape(1,kodex1_shape,kodex2_shape)


y_predict = model.predict([x_predict,kodex_x_predict])


print(y_predict)

#loss, mae:  9997190.0 2357.630126953125
#RMSE:  3161.833
#R2:  0.8881030596933368
#[[89722.375 89887.71 ]]