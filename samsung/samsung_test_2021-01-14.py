#기존 데이터

import numpy as np
import pandas as pd


# 데이터 변수
size = 30 #30


# 데이터 불러오기 
df = pd.read_csv('../data/삼성전자.csv', index_col=0,header=0,encoding='CP949')
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')


df1 = pd.read_csv('./samsung/삼성전자2.csv', index_col=0,header=0,encoding='CP949')
df1.replace(',','',inplace=True, regex=True)


#df1 = df1.iloc[0,[0,1,2,7,8,3]]
# 액분 전 데이터
df1 = df1.iloc[0,[0,1,2,7,10,3]]
# 액분 후 데이터


df1 = df1.astype('float32')


# 액분 전 데이터
'''
df = df.iloc[:662,:]
df.drop(['등락률', '기관' ,'프로그램','신용비','개인','외인(수량)','외국계','외인비'], axis='columns', inplace=True)
'''
# 상관 관계 50 먹이기 전 (거래량, 가격)



# 액분 후 데이터 

df_1 = df.iloc[:662,:]
df_2 = df.iloc[665:,:]
df = pd.concat([df_1,df_2])
df.iloc[662:,0:4] = df.iloc[662:,0:4]/50.0
df.iloc[662:,5:] = df.iloc[662:,5:]*50
df.drop(['등락률', '기관' ,'금액(백만)','신용비','프로그램','외인(수량)','외국계','외인비'], axis='columns', inplace=True)

# 상관 관계 50 먹이기 후 (거래량, 개인)


df = df.sort_values(by=['일자'], axis=0)



df_x = df.iloc[:,[0,1,2,4,5,3]]

df_x = df_x.append(df1)


total_data = df_x.to_numpy()


def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

total_data = split_x(total_data,size)



# npy 저장
np.save('./samsung/samsung_data.npy', arr=total_data)

# npy 불러오고 데이터 처리 후 모델 

total_data = np.load('./samsung/samsung_data.npy')

x = total_data[:-2,:size, :-1]
y = total_data[2:,size-1,-1:]
x_pred = total_data[-2,:size,:-1]

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


#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,Input,Activation, LSTM

input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
dense1 = LSTM(150)(input1)#145
dense1 = Dense(200, activation = 'relu')(dense1)
dense1 = Dense(150, activation = 'relu')(dense1)
dense1 = Dense(150, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dense(25, activation = 'relu')(dense1)
dense1 = Dense(10, activation = 'relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3, verbose=1, callbacks=[early_stopping])


model.save('./samsung/samsung_model.h5')


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

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()


# loss, mae:  2531887.5 1221.1883544921875
# RMSE:  1591.1909
# R2:  0.9657210086094726
# [[88653.445]]


# 모든 데이터 예측 

# loss, mae:  2793302.75 1398.217041015625
# RMSE:  1671.3175
# R2:  0.9839767439755686
# [[85331.66]]

# 액분 후 데이터 예측

# loss, mae:  2446270.5 1174.77734375
# RMSE:  1564.0563
# R2:  0.9699761592040256
# [[89992.6]]