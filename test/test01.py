
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.           # 최대값인 255를 나누었으니 최소0~ 최대1이 된다. *float32이란 실수로 바꾼다는 뜻
x_test = x_test.reshape(10000, 28, 28, 1)/255.                             # 실수형이라는 것을 빼도 인식한다.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))      # 실제로 코딩할 때는 이 방법이 가장 좋다!
 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# y 리쉐잎
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# y 벡터화 OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()
y_val = hot.transform(y_val).toarray()

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=120, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(100, 2, strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(80, 2, strides=1))
model.add(Flatten())
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# 컴퓨터 시간 보여주는 함수 
import datetime 
date_now = datetime.datetime.now()
print(date_now)
date_time = date_now.strftime("%m%d_%H%H_%M%M_%S%S")
print(date_time)

filepath = '../data/modelcheckpoint/'
filename = '_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath,"k45_",date_time,filename])


# d =  정수형으로 10의 자리까지 /f = float 실수형으로 소수 4번째까지 하겠다. ##이부분 찾아보기

stop = EarlyStopping(monitor='val_loss', patience=10, mode='min') #monitor=val_loss도 가능, 그냥 loss보다 val_loss를 신뢰하기도 한다.
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# 파일패스는 파일형태로 기록한다는 뜻이다. 최저점마다 파일을 생성하는데 파일을 만들어 주는 이유는 그 지점의 w를 저장하기 떄문이다. 
# w가 저장되어 있으면 훈련을 다시 할 필요가 없으니까
# 마지막 파일이 가장 좋으니 나머지는 삭제하고 마지막 파일은 가중치를 불러와서 쓸 수 있다.
# k45_mnist_15-0.0554 > 45번 파일의 15번째 훈련이 val_loss가 0.0554로 가장 낮다. 
# 모델이 완벽하다는 가정하에 최적의 기록이다. 튜닝을 확실히 해놓을 것

hist = model.fit(x_train, y_train, epochs=30, batch_size=28, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, mc])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=28)
print('loss, acc: ', loss, acc)

