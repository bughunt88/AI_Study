from numpy import genfromtxt
from matplotlib import pyplot

import matplotlib.image

from matplotlib.image import imread
import pandas as pd

# train 데이터 (1회 대회 train, test 데이터에서 끌어온다)
train_a = pd.read_csv('../data/vision2/mnist_data/test.csv')
train = pd.read_csv('../data/vision2/mnist_data/train.csv')

# *********************
# train 데이터 
# 1회에 쓰던 mnist 데이터 A를 모아서 128으로 리사이징 해준다 
# 알파벳 별로 모델을 만들 것!

tmp1 = pd.DataFrame()

train = train.drop(['id','digit'],1)
train_a = train_a.drop(['id'],1)

tmp1 = pd.concat([train,train_a])

tmp1.loc[tmp1['letter']!='A','letter'] = 0
tmp1.loc[tmp1['letter']=='A','letter'] = 1


x_train = tmp1.to_numpy().astype('int32')[:,1:] # (852, 784)
y_train = tmp1.to_numpy().astype('int32')[:,0] # (852,)


# 이미지 전처리 100보다 큰 것은 254으로 변환, 100보다 작으면 0으로 변환
x_train[100 < x_train] = 254
x_train[x_train < 100] = 0

x_train = x_train.reshape(-1,28,28,1)

my_data = x_train[0]

import numpy as np
import PIL
import matplotlib.pyplot as plt
# load the data
plt.figure
plt.imshow(my_data)

plt.show()

# finally save the image as jpg file
image = PIL.Image.fromarray(my_data, 'RGB')
image.save('../data/vision2/cut/1.png')



