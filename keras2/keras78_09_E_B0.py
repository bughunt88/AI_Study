from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.optimizers import Adam

import numpy as np

transfer = EfficientNetB0(weights='imagenet', include_top=False,input_shape=(32,32,3))

transfer.trainable = False
'''
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
'''

# vgg16.trainable = True
'''
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
'''
(x_train, y_train), (x_test,y_test)= cifar10.load_data()


# 데이터 전처리

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# x 같은 경우 색상의 값이기 때문에 255가 최고 값


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM
model = Sequential()
model.add(transfer)
# model.add(Conv2D(filters=1024,kernel_size=(1,1),padding='valid')) # 미세조정(파인튜닝)
# model.add(Conv2D(filters=1024,kernel_size=(1,1),padding='valid')) # 미세조정(파인튜닝)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

# print("그냥 가중치의 수 : ", len(model.weights))   #32 -> (weight가 있는 layer * (i(input)bias + o(output)bias))
# print("동결 후 훈련되는 가중치의 수 : ",len(model.trainable_weights))   #6
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
modelpath = './modelCheckpoint/k46_cifa10_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience= 20)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
lr = ReduceLROnPlateau(factor=0.1,verbose=1,patience=10)

model.compile(loss = 'categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
history =model.fit(x_train,y_train, epochs=100, batch_size=120, validation_split=0.2,  
                                     callbacks = [early_stopping,cp])

#4. evaluate , predict

loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)


y_predict = model.predict(x_test)
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_test : ",y_test[:10])
print("y_test : ",y_test[:10])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
plt.plot(history.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
plt.show()