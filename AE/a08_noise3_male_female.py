import numpy as np
from tensorflow.keras.datasets import mnist

x_train = np.load('../data/image/sex/numpy/keras67_train_x.npy')
y_train = np.load('../data/image/sex/numpy/keras67_train_y.npy')
x_test = np.load('../data/image/sex/numpy/keras67_test_x.npy')
y_test = np.load('../data/image/sex/numpy/keras67_test_y.npy')

x_train_noised = x_train + np.random.normal(0,0.2, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.2, size=x_test.shape)
x_trina_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D

def autoencoder():
    model = Sequential()
    model.add(Conv2D(256, 3, activation= 'relu', padding= 'same', input_shape = (150,150,3)))
    model.add(Conv2D(256, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid'))

    return model

model = autoencoder() 
# 95%의 pca 수치가 154
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5, figsize=(20,7))


# 이미지 다셧개를 무작위로 고른다. 

random_image =random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다!!!

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_image[i]].reshape(150,150,3), cmap='gray')  
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지 

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_image[i]].reshape(150,150,3), cmap='gray')  
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래 그린다.

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_image[i]].reshape(150,150,3), cmap='gray')  
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
