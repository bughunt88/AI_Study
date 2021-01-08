

import numpy as np

from sklearn.datasets import load_boston


# 1. 데이터

dataset  = load_boston()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)
print("============")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))
print(dataset.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 )


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)




# 2. 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout


# model.add(Dropout(0.2))
# 모든 레이어에 쓸 수 있다 
# 레이어 아래 써야 위 레이어에 적용된다 

# Dropout을 사용하면 모든 곳에 다 먹힌다 
# 직접 수정하면 훈련에만 적용된다 


model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 13)) 
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(1))   

# 함수형의 사용법
# dense1 = Dense(5, activation='relu')(input1)
# dense1 = Dropout(0.2)(dense1)




# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')



from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=20, mode='min') # mode는 min,max,auto 있다

model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])




# 4. 평가, 예측



loss, mae = model.evaluate(x_test, y_test, batch_size=8)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

print('loss : ', loss)
print('mae : ', mae )
print("RMSE : ", RMSE(y_test, y_predict))

# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# 제대로 전처리 (validateion_split)
# loss :  10.961244583129883
# mae :  2.407649278640747
# RMSE :  3.3107768583643407
# R2 :  0.8673247641902566


# 제대로 전처리 (validateion_data)
# loss :  8.83780288696289
# mae :  2.1395821571350098
# RMSE :  2.972844440966045
# R2 :  0.8930269408648741

# dropout 사용시
# loss :  10.941795349121094
# mae :  2.1144745349884033
# RMSE :  3.307838427310614
# R2 :  0.8675601675675978