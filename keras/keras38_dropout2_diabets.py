
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target


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

model=Sequential()
model.add(Dense(128, activation='relu', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')



# EarlyStopping 

from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=2, mode='auto') # mode는 min,max,auto 있다

model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


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
