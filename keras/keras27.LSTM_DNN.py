# keras23_LSTM3_scale 을 DNN으로 코딩
# 결과치 비교

# DNN으로 23번 파일보다 loss를 좋게 만들기

import numpy as np


# 1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])


y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_pred = np.array([50,60,70])


# print(x.shape) # (13,3)
# print(y.shape) # (13,)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 결과값 나오는 것도 transform 해줘야 한다!
x_pred = x_pred.reshape(1,3) # 3차원 
x_pred = scaler.transform(x_pred)



# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=3)) 
# model.add(Dense(50, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))


model.add(Dense(1))



# 3. 컴파일 훈련


model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping
# eraly_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') 
#model.fit(x_train, y_train, epochs=250, batch_size=20, validation_data=(x_val,y_val), callbacks=[eraly_stopping])

model.fit(x_train, y_train, epochs=250, batch_size=20, validation_data=(x_val,y_val))


# 4. 평가, 예측

loss = model.evaluate(x_test, y_test, batch_size=20)
print(loss)
y_predict = model.predict(x_pred)
# 지표를 만들기 위한 프레딕트 

print(y_predict)

# keras23_LSTM3_scale 결과
# 0.10277651250362396
# [[80.14559]]

# keras27.LSTM_DNN 결과
# 0.0207461379468441
# [[80.39788]]