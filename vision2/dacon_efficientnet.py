
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import string
from tensorflow.keras.models import load_model

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import decode_predictions
import efficientnet.keras as efn 


import os
'''
alphabets = string.ascii_lowercase
alphabets = list(alphabets)
'''

'''
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
'''

model = efn.EfficientNetB0(weights='imagenet')

#image = imread('../data/vision2/cat.jpg')
image = imread('../data/vision2/test_dirty_mnist_2nd/50001.png')

image_size = model.input_shape[1] # 224

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

x = efn.center_crop_and_resize(image, image_size=image_size)

plt.figure(figsize=(10, 10))
plt.imshow(x)
plt.show()

print(x.shape)

x = efn.preprocess_input(x)

print(x.shape)


x = np.expand_dims(x, 0)

x = x.reshape(x.shape[0],x.shape[1],x.shape[2],3)




print(x.shape)


y = model.predict(x)

dy = decode_predictions(y)

print(dy)
'''
def inference(model, image_path):
    image = imread(image_path)
    
    image_size = model.input_shape[1] # 224
    cx = center_crop_and_resize(image, image_size=image_size)
    
    x = preprocess_input(cx)
    x = np.expand_dims(x, 0)

    y = model.predict(x)
    dy = decode_predictions(y)[0]
    
    for idx, label, confidence in dy:
        print('%s: %.2f%%' % (label, confidence * 100))
    
    plt.figure(figsize=(5, 5))
    plt.imshow(cx.astype(np.uint8))
    plt.show()


inference(model, 'imgs/01.jpg')
'''