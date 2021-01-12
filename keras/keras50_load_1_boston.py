
# EarlyStopping


import numpy as np

x = np.load('../data/npy/boston_x.npy')
y = np.load('../data/npy/boston_y.npy')


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)




# 2. 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 13)) 

model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))   # 아웃풋 결과가 달라지면 여기도 수정해야한다 



# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')





# EarlyStopping 
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

