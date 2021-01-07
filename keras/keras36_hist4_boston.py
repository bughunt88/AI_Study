# hist를 이용하여 그래프를 그리시오.
# loss, val_loss


import numpy as np

#1 데이터 주고
from tensorflow.keras.datasets import boston_housing
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

print(x_train.shape) #(404, 13)
#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(70, activation='relu')(input1)
dense1 = Dense(140, activation='relu')(dense1)
dense1 = Dense(140, activation='relu')(dense1)
dense1 = Dense(140, activation='relu')(dense1)
dense1 = Dense(70, activation='relu')(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')
hist = model.fit(x_train, y_train, epochs=2000, batch_size=4, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
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

#1번파일
# loss, mae:  11.377790451049805 2.260841131210327
# RMSE:  3.3730985125849697
# R2:  0.8633197073667653

#2번파일 
# loss, mae:  23.424747467041016 3.4580018520355225
# RMSE:  4.8399119673460556
# R2:  0.7186008543791937


# 그래프 적용
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss & val_loss')
plt.ylabel('loss, val_loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()