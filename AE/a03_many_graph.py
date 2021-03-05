import numpy as np
from tensorflow.keras.datasets import mnist

# 오토인코더 - 비지도 학습, 차원축소에도 사용
# y값이 없다!!
# 784, 엠니스트 데이터가 64 덴스레이어로 들어가고 다시 784, 로 나온다면
# 데이터 축소, 데이터 확장이 같이 이뤄진다

(x_train, _), (x_test, _) = mnist.load_data() # _ 은 쓰지 않을 변수, y 값을 사용하지 않을것이다!

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.

# print(x_train[0]) # 잘출력되는지 확인
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, activation = 'relu', input_shape = (784,)))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

outputs = [x_test]

for i in range(6):

    a = 2**i
    print(f'node {a}개 시작')
    model = autoencoder(hidden_layer_size=a)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    model.fit(x_train, x_train, epochs = 10)

    outputs.append(model.predict(x_test))

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize = (15,15))

random_imgs = random.sample(range(outputs[0].shape[0]), 5)

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28),
                cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()