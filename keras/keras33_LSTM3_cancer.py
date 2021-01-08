# 사이킷런 데이터셋
# LSTM으로 모델링
# Dense 와 성능비교



import numpy as np
from sklearn.datasets import load_breast_cancer


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) 
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) 
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1) 


# 2. 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(120, activation='relu', input_shape=(30,1)))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1, activation='sigmoid')) 
# sigmoid는 0~1 사이로 한정을 하는 코드
# 히든이 없는 모델도 가능하다 


# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


loss, mae = model.evaluate(x_test, y_test, batch_size=8)



y_predict = model.predict(x_val)

print('y_predict: ', y_predict)
# [[1.0000000e+00 8.6358709e-10]
#  [4.4308349e-02 9.5791960e-01]
#  [1.0000000e+00 3.0873505e-08]
#  [1.0034841e-02 9.9010718e-01]]

print('y_predict_argmax: ', y_predict.argmax(axis=1)) #0이 열, 1이 행




# 그래프 
import matplotlib.pyplot as plt


# plt.plot(x,y)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val loss', 'train acc', 'val acc'])
plt.show()
