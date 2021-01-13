
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
from tensorflow.keras.layers import Dense, Input, LSTM,Conv1D,MaxPooling1D,Dropout,Flatten

model=Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(30,1))) 
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=52, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
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

print('loss : ', loss)
print('mae : ', mae)


# loss :  0.15454581379890442
# mae :  0.9532163739204407 


# conv1d
# loss :  0.14320489764213562
# mae :  0.9532163739204407


