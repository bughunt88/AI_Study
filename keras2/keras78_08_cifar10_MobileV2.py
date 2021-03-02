from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10
from keras.applications.mobilenet_v2 import preprocess_input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy as np

transfer = MobileNetV2(weights='imagenet', include_top=False,input_shape=(32,32,3),alpha=1.4)
# alpha는 모바일넷에 있는 파라미터 중 하나로 0.3, 0.5 0.75, 1.0, 1.3, 1.4로 조정이 가능합니다.
# alpha : 네트워크의 너비를 제어합니다. 이것은 MobileNet 논문에서 폭 승수로 알려져 있습니다.
# alpha 1.0 미만 이면 각 레이어의 필터 수를 비례 적으로 감소시킵니다.
# alpha 1.0을 초과하면 각 층 필터의 개수를 비례 적으로 증가시킨다. 
# alpha= 1이면 용지의 기본 필터 수가 각 레이어에 사용됩니다. 기본값은 1.0입니다.
# 1로 했을 때 가장 안좋은 결과를 얻었는데... 이유는 아직까진 잘 모르겠습니다. 1.4적용 => 정확도 0.46



# transfer.trainable = False



(x_train, y_train), (x_test,y_test)= cifar10.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size = 0.8)

# 데이터 전처리

x_train = preprocess_input(x_train) # mode = tf
x_test = preprocess_input(x_test)
x_val = preprocess_input(x_val)
# x_train = x_train/255.
# x_test = x_test/255.

print(x_train)
#    [-0.56078434 -0.42745095 -0.6627451 ]
#    [-0.5294118  -0.40392154 -0.6313726 ]
#    [-0.5294118  -0.41960782 -0.6313726 ]]]]



# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM
model = Sequential()
model.add(transfer)
# model.add(Conv2D(filters=1024,kernel_size=(1,1),padding='valid')) # 미세조정(파인튜닝)
# model.add(Conv2D(filters=1024,kernel_size=(1,1),padding='valid')) # 미세조정(파인튜닝)
# model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation= 'softmax'))

transfer.summary()
# model.summary()

for i, layer in enumerate(transfer.layers):
       print(i, layer.name)
# ...
# 147 block_16_depthwise
# 148 block_16_depthwise_BN
# 149 block_16_depthwise_relu
# 150 block_16_project
# 151 block_16_project_BN
# 152 Conv_1
# 153 Conv_1_bn
# 154 out_relu

for layer in transfer.layers[:152]: # 151 레이어층까지 가중치를 동결시키고
   layer.trainable = False
for layer in transfer.layers[152:]: # 152 레이어부터는 가중치를 갱신시키겠다.
   layer.trainable = True
# 다소 과적합은 있었으나 0.48 -> 0.64까지 올랐다...

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
modelpath = './modelCheckpoint/k46_cifa10_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience= 10)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
lr = ReduceLROnPlateau(factor=0.5,verbose=1,patience=5)

model.compile(loss = 'categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
history =model.fit(x_train,y_train, epochs=100, batch_size=64, validation_data = (x_val,y_val) , 
                                     callbacks = [early_stopping,cp])

#4. evaluate , predict

loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)


y_predict = model.predict(x_test)
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_test : ",y_test[:10])
print("y_test : ",y_test[:10])

# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
# plt.plot(history.history['val_acc'])
# plt.title('loss & acc')
# plt.ylabel('loss & acc')
# plt.xlabel('epoch')
# plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
# plt.show()
