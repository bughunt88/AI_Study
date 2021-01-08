
import numpy as np
from sklearn.datasets import load_breast_cancer


# 1. 데이터 

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


# 2. 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(120, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(120))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 


# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다

model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


loss= model.evaluate(x_test, y_test, batch_size=8)
print(loss)

y_pred = model.predict(x_test)

array_list = []

for n in model.predict(x_test):

    if n >= 0.5:
        array_list.append(1)
    else:
        array_list.append(0)


print(y_pred[np.where(y_pred > 0.5)])