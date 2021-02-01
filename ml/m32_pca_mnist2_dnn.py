# m31로 만든 1.0 이상의 n_component = ? 를 사용하여 dnn 모델을 만들 것

# mnist dnn 보다 성능 좋게 만들어라!
# cnn과 비교 

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA


(x_train, y_train), (x_test,y_test)= mnist.load_data()

# plt.imshow(x_train[1])
# plt.show()


x = np.append(x_train, x_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])


pca1 = PCA()
pca1.fit(x)
cumsum = np.cumsum(pca1.explained_variance_ratio_)

pca_num = np.argmax(cumsum >= 0.95)+1

pca = PCA(n_components=pca_num)
x2 = pca.fit_transform(x)

# 데이터 전처리


x_train, x_test, y_train, y_test = train_test_split(x2, y,  train_size=0.7, random_state = 77, shuffle=True ) 

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)

from tensorflow.keras.utils import to_categorical # OneHotEncoding from tensorflow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



from tensorflow.keras.models import Sequential,  Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM, Input


inputs = Input(shape=(pca_num,))

dense1 = Dense(128, activation='relu')(inputs)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(48, activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dropout(0.2)(dense1)

outputs = Dense(10,activation='softmax')(dense1)

model = Model(inputs= inputs, outputs = outputs)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=80, validation_split=0.2, verbose=1, callbacks=[stop])


#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=80)
y_predict = model.predict(x_test[:10])

print(loss)
print(mae)

print('y_test : ',  y_test[:10].argmax(axis=1))
print('y_predict_argmax : ', y_predict.argmax(axis=1)) 



# CNN
# 0.02563497982919216
# 0.991599977016449
# y_test :  [7 2 1 0 4 1 4 9 5 9]
# y_predict_argmax :  [7 2 1 0 4 1 4 9 5 9]


# DNN
# 0.13371659815311432
# 0.9804999828338623
# y_test :  [7 2 1 0 4 1 4 9 5 9]
# y_predict_argmax :  [7 2 1 0 4 1 4 9 5 9]

# PCA DNN
# 0.1436588168144226
# 0.9718571305274963
# y_test :  [7 4 3 7 7 1 9 6 6 6]
# y_predict_argmax :  [7 4 3 7 3 1 9 6 6 6]