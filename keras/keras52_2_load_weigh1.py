
# load_weights, load_model

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 

#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model =Sequential()
model.add(Conv2D(filters=50, kernel_size=(3,3), 
                padding='same', strides=(1,1), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(5))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2)))
model.add(Flatten())
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



model.load_weights('../data/h5/k2_1_weigth.h5')
# 가중치 세이브 불러오기 (모델은 저장 안됨)
model2 = load_model('../data/h5/k52_1_model2.h5')
# 훈련을 저장했기 때문에 데이터만 있으면 된다



#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('가중치_loss : ', result[0])
print('가중치_acc : ', result[1])

# 가중치_loss :  0.07213222980499268
# 가중치_acc :  0.9775999784469604


result2=model2.evaluate(x_test,y_test, batch_size=16)
print('로드모델_loss : ', result2[0])
print('로드모델_acc : ', result2[1])

# 로드모델_loss :  0.07213222980499268
# 로드모델_acc :  0.9775999784469604

