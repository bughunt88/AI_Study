import io
import pandas as pd
import numpy as np

size = 5 #30

# 삼성 데이터 
filename = '../data/samsung/삼성전자.csv'
filename2 = '../data/samsung/삼성전자2.csv'
filename3 = '../data/samsung/삼성전자3.csv'

df = pd.read_csv(filename, index_col=0,header=0,encoding='CP949')
df2 = pd.read_csv(filename2, index_col=0,header=0,encoding='CP949')
df3 = pd.read_csv(filename3, index_col=0,header=0,encoding='CP949')

df.replace(',','',inplace=True, regex=True)
df2.replace(',','',inplace=True, regex=True)
df3.replace(',','',inplace=True, regex=True)

df = df.iloc[:662,[0,1,2,6,3]]
df2 = df2.iloc[0,[0,1,2,8,3]]
df3 = df3.iloc[0,[0,1,2,8,3]]

df = df.astype('float32')
df2 = df2.astype('float32')
df3 = df3.astype('float32')

df = df.sort_values(by=['일자'], axis=0)
df = df.append(df2)
df = df.append(df3)

total_data = df.to_numpy()
y_data = total_data

np.save('../data/samsung_data.npy', arr=total_data)


# 코덱스 데이터
kodex_data = pd.read_csv('../data/samsung/kodex.csv', index_col=0,header=0,encoding='CP949')
kodex_data.replace(',','',inplace=True, regex=True)


kodex_data.drop(['전일비'], axis='columns', inplace=True)
kodex_data = kodex_data.astype('float32')
kodex_data = kodex_data.iloc[:664,:]
kodex_data = kodex_data.iloc[:,[0,1,2,6,7,3]]
kodex_data = kodex_data.sort_values(by=['일자'], axis=0)

total_kodex_data = kodex_data.to_numpy()

np.save('../data/samsung/kodex_data.npy', arr=total_kodex_data)


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
x_pred = total_data[-2,:size,:]


# print(x_pred)
print(x[-1])
print(y[-1])

x1_shape = x.shape[1]
x2_shape = x.shape[2]

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred = x_pred.reshape(1,x_pred.shape[0]* x_pred.shape[1])



# 코덱스 데이터 
kodex_x = total_kodex_data[:-2,:size, :]
kodex_x_pred = total_kodex_data[-2,:size,:]

kodex1_shape = kodex_x.shape[1]
kodex2_shape = kodex_x.shape[2]

#print(kodex_x_pred)
#print(kodex_x[-1])
#print(kodex_x_pred)

kodex_x = kodex_x.reshape(kodex_x.shape[0], kodex_x.shape[1]*kodex_x.shape[2])
kodex_x_pred = kodex_x_pred.reshape(1,kodex_x_pred.shape[0]* kodex_x_pred.shape[1])


# 데이터 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, kodex_train, kodex_test = train_test_split(x, y, kodex_x, train_size=0.7, shuffle=False)

# 전처리
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



#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,Input, LSTM, concatenate



#model = Sequential()
'''
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(32, activation='relu'))
#model.add(LSTM(16,return_sequences=True,activation='softsign'))
model.add(LSTM(8, activation='relu'))
model.add(Dense(2))
'''


input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))
model1 = Conv1D(filters=16, kernel_size=1, activation='relu')(input1)
model1 = MaxPooling1D(pool_size=2)(model1)
model1 = LSTM(8, activation='relu')(model1)
#model1 = Dense(2)(model1)

input2 = Input(shape=(kodex_train.shape[1],kodex_train.shape[2]))
model2 = Conv1D(filters=16, kernel_size=1, activation='relu')(input2)
model2 = MaxPooling1D(pool_size=2)(model2)
model2 = LSTM(8, activation='relu')(model2)
#model2 = Dense(2)(model2)


# concatenate
merge1 = concatenate([model1, model2])
model3 = Dense(32, activation='relu')(merge1)
model3 = Dense(16, activation='relu')(model3)




# output
output1 = Dense(2)(model3)   # y1 :output =  2 (마지막 아웃풋)

model = Model(inputs=[input1,input2], outputs=output1)


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

savepath = '../data/samsung/check_point.h5'

cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)


print(x_train.shape)
print(kodex_train.shape)
print(y_train.shape)



hist = model.fit([x_train,kodex_train], y_train, epochs=500,  validation_split=0.3, verbose=1, batch_size=4 ,callbacks=[early_stopping, cp,reduce_lr])


# model.save('./samsung/samsung_model.h5')


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



import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()


#loss, mae:  9997190.0 2357.630126953125
#RMSE:  3161.833
#R2:  0.8881030596933368
#[[89722.375 89887.71 ]]