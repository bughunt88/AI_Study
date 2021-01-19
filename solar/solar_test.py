

import numpy as np
import pandas as pd


df = pd.read_csv('../data/solar/train.csv', index_col=None,header=0,encoding='CP949')

df = df.astype('float32')

total_data = df.to_numpy()

def split_xy(dataset, time_steps, y_column, x_steps):
    x,y = list(), list()
    for i in range(len(dataset)):

        i = i*time_steps

        x_end_number = i + time_steps*x_steps
        y_end_number = x_end_number + time_steps*y_column

        if y_end_number > len(dataset):
            break

        temp_x = dataset[i:x_end_number,:]
        temp_y = dataset[x_end_number:y_end_number]

        x.append(temp_x)
        y.append(temp_y)

    return np.array(x), np.array(y)

x, y = split_xy(total_data,336,2,7)



y_index_list = y[0,:,:3]

# x, y의 실 데이터 정리 
x = x[:,:,3:]
y = y[:,:,3:]



x_shape1 = x.shape[1]
x_shape2 = x.shape[2]
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)

x_train = x_train.reshape(x_train.shape[0], x_shape1*x_shape2)
x_test = x_test.reshape(x_test.shape[0], x_shape1*x_shape2)
'''

y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])


'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_shape1, x_shape2)
x_test = x_test.reshape(x_test.shape[0], x_shape1, x_shape2)
'''



#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,Input,Activation, LSTM, Reshape, MaxPool1D


model = Sequential()

#model.add(LSTM(10, activation='relu', input_shape=(336,9))) 
#model.add(Reshape(target_shape=(96, 10)))


model = Sequential()
model.add(Conv1D(filters=32, kernel_size=1, activation='relu',input_shape=(336,6)))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(96*6))

'''
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=1, activation='relu',input_shape=(336,9)))
model.add(MaxPool1D(pool_size=2))
model.add(LSTM(8, activation='relu'))
model.add(Dense(96*9))
'''


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

#savepath = './samsung/check_point.h5'

#cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

hist = model.fit(x, y, epochs=1,  validation_split=0.3, verbose=1, batch_size=32 ,callbacks=[early_stopping]) #, cp])

#평가, 예측
loss, mae = model.evaluate([x], y, batch_size=32)
print('loss, mae: ', loss, mae)

y_predict = model.predict([x])
y_predict = y_predict.reshape(y_predict.shape[0], 96, 6)


total_y_predict = []
total_predict = []

for n in range(81):
    file_title = '../data/solar/test/'+ str(n) + '.csv'
    load_data = pd.read_csv(file_title, index_col=None,header=0,encoding='CP949')
    load_data = load_data.to_numpy()
    load_data_x = load_data[:,3:]
    load_data_x_res = load_data_x.reshape(1, load_data_x.shape[0],load_data_x.shape[1])
    #모델에서 predict 값 넣는다
    load_predict = model.predict(load_data_x_res)
    load_predict = load_predict.reshape(96,6)
    total_predict.append(load_predict)

    total_y_predict.append(y_index_list)




total_predict = np.array(total_predict)
total_predict = total_predict.reshape(total_predict.shape[0]*total_predict.shape[1], 6)
total_predict = pd.DataFrame(total_predict)

print(total_predict.shape)

total_y_predict = np.array(total_y_predict)
total_y_predict = total_y_predict.reshape(total_y_predict.shape[0]*total_y_predict.shape[1], 3)

print(total_y_predict.shape)

total_predict = pd.concat([total_predict, total_predict])

print(total_predict.shape)


#total_predict.to_csv('../data/solar/total_predic.csv', sep=',')




#     file_title =''

#     print(load_data)



# print(y_predict) # (1087, 96, 9)
# print(y_predict.shape) # (1087, 96, 9)







'''
#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y, y_predict)
print('R2: ', R2)

'''