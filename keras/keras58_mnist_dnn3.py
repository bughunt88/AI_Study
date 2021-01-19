# 인풋 4차원 아웃풋 4차원 출력 



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.



y_train = x_train
y_test = x_test

print(y_train.shape)
print(y_test.shape)



#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model =Sequential()
#model.add(Conv2D(filters=64, kernel_size=(3,3), 
#               padding='same', strides=(1,1), input_shape=(28,28,1)))
#model.add(MaxPooling2D(pool_size=2))

model.add(Dense(64, input_shape=(28,28,1)))
model.add(Dropout(0.5))
#model.add(Conv2D(1, (2,2)))
#model.add(Conv2D(1, (2,2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1)) 
model.summary()



#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=10)
# cp =ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.5,callbacks=[es,reduce_lr])

#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])


y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)