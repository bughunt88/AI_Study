# 다차원 댄스 모델
# (n,32,32,3) -> (n,10)

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train=x_train.astype('float32')/255.
x_test= x_test/255.
print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(10000, 32, 32, 3)
#이미지 특성맞춰 숫자 바꾸기 x의 최대가 255이므로 255로 나눈다.
#이렇게 하면 0~1 사이로 된다.
#x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],-1)
#x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],-1)
###이미 전처리 이걸로 해서 minmax안써도 됨

#다중분류 y원핫코딩
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train) #(50000, 10)
y_test = to_categorical(y_test)  #(10000, 10)

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

Xception = Xception(weights = 'imagenet', include_top=False, input_shape=(32,32,3))

Xception.trainable = False
# 훈련을 시킨다


#2. 모델 구성
#layer문을 for문으로 써도 된다. #여기는 y가 2차원이니까 Flatten하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
model=Sequential()

model.add(Xception)
model.add(Flatten())

#model.add(Dense(units=12, activation='relu', input_shape=(32,32,3)))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15,activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
           
model.fit(x_train, y_train, epochs=200, batch_size=8, validation_split=0.2, verbose=1, callbacks=[es,lr])


#4. 평가 훈련
loss=model.evaluate(x_test,y_test, batch_size=16)
print(loss)

'''
dnn
[2.3026492595672607, 0.10000000149011612]
y_pred:  [6 6 6 6 6 6 6 6 6 6]
y_test:  [3 8 8 0 6 6 1 6 3 1]
4차원 결과
[1.5934568643569946, 0.4643999934196472]
'''