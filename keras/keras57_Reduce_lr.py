# 45번 카피해서 56에 복붙!!!


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
model.add(Conv2D(filters=64, kernel_size=(3,3), 
                padding='same', strides=(1,1), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv2D(1, (2,2)))
model.add(Conv2D(1, (2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
#model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=10)
# cp =ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto')




# 기울기를 줄여서 훨신 좋게 해준다 (역전파) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)




model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.5,callbacks=[es,reduce_lr])

#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

####시각화
import matplotlib.pyplot as plt

plt.rc('font', family='NanumGothic') # For Windows

plt.figure(figsize=(10,6)) #가로세로 #이거 한 번 만 쓰기
plt.subplot(2,1,1)   #서브면 2,1,1이면 2행 1열 중에 1번째라는 뜻
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() #바탕을 grid모눈종이로 하겠다

plt.title('손실비용') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend('accuracy','val_accuracy')

plt.show()