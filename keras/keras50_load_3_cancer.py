import numpy as np
from sklearn.datasets import load_breast_cancer


# 1. 데이터 

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


# 2. 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(120, activation='relu', input_shape=(30,)))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1, activation='sigmoid')) 


# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다

model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])

loss= model.evaluate(x_test, y_test, batch_size=8)
print(loss)


y_pred = model.predict(x_test)
# 이진은 아래 코드를 써서 데이터 값을 확인한다 
print(y_pred[np.where(y_pred > 0.5)])