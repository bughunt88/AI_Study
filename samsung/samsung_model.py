
import numpy as np
import pandas as pd


# 데이터 변수
size = 30 #30


total_data = np.load('./samsung/samsung_data.npy')

x = total_data[:-1,:size, :-1]
y = total_data[1:,size-1,-1:]
x_pred = total_data[-1,:size,:-1]

x1_shape = x.shape[1]
x2_shape = x.shape[2]

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred = x_pred.reshape(1,x_pred.shape[0]* x_pred.shape[1])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x1_shape, x2_shape)
x_test = x_test.reshape(x_test.shape[0], x1_shape, x2_shape)
 
from tensorflow.keras.models import load_model

# 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,Input,Activation, LSTM

model = load_model('./samsung/samsung_model_v2.h5')


#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)


x_predict = x_pred.reshape(1,x1_shape,x2_shape)
y_predict = model.predict(x_predict)

print(y_predict)

'''
loss, mae:  667091.5625 621.8015747070312
RMSE:  816.75684
R2:  0.9962432600121626
[[89632.67]]
'''


