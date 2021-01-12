import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.

#다중분류 y원핫코딩
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
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
model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
#model.summary()



model.save('../data/h5/k52_1_model1.h5')
# 모델만 저장


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k52_1_MCK_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5)
cp =ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=1, validation_split=0.2,callbacks=[es,cp])




model.save('../data/h5/k52_1_model2.h5')
# 훈련한 다음에 모델을 저장을 하게 되면 가중치(W) 까지 저장된다!
# 모델을 저장하고 싶으면 컴파일 훈련 전에 저장
# 가중치까지 저장하고 싶으면 컴파일 훈련 후 저장 

model.save_weights('../data/h5/k2_1_weigth.h5')
# 웨이트 저장


#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])
