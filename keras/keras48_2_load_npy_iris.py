import numpy as np

x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)


import numpy as np
from sklearn.datasets import load_iris


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,  train_size=0.7, random_state = 66 ) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3,  random_state = 66 ) 


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


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
model.add(Dense(169, input_shape=(4,), activation='relu'))
model.add(Dense(169, activation='relu'))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다

model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


loss= model.evaluate(x_test, y_test, batch_size=8)
print(loss)


print(x_test[-5:-1])
y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])



y_predict=np.argmax(y_pred, axis=1)
print('예측값 : ', y_predict)
