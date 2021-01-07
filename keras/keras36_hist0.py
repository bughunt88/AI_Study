import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 101))
size = 5 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):     #행
        subset = seq[i : (i+size)]           #열
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset.shape) #(96, 5)

x = dataset[:, 0:4]
y = dataset[:, -1]
print(x.shape, y.shape)     #(96, 4) (96,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              #(96, 4, 1)

#2. 모델
model = load_model('./model/save_keras35.h5')
model.add(Dense(5, name = 'new1'))
model.add(Dense(1, name = 'new2'))

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=10, mode='auto')

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[stop])

print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001EB02BBB370>
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])           # model.fit에서 반환하는 것

print(hist.history['loss'])
# [2258.907958984375, 1968.247314453125, 1636.9425048828125,                            # loss의 이력이 나온다. 이걸 시각화 해보자.
# .... 0.005746894981712103, 0.002807446289807558]


# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
# plt.plot(x,y) 이걸 넣어도 들어간다

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['loss', 'val loss', 'acc', 'val acc']) #그래프이름
plt.show() 

# A // loss: 0.09 , val_loss: 0.9
# B // loss : 0.9,  val_loss: 0.91
# 일 때 A는 차이가 너무 커 과적합 상태이다. 따라서 B모델을 더 좋은 모델로 본다.
# 간혹 loss와 val_loss의 차이보다 val_loss의 값이 얼마나 작냐로 모델을 평가하기도 한다.