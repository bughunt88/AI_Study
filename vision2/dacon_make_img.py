
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
from PIL import Image

train_a = pd.read_csv('../data/vision2/mnist_data/test.csv')
train = pd.read_csv('../data/vision2/mnist_data/train.csv')

print("@@@@@@@@@@@@@")
print(type(train))
# *********************
# train 데이터 
# 1회에 쓰던 mnist 데이터 A를 모아서 128으로 리사이징 해준다 
# 알파벳 별로 모델을 만들 것!

tmp1 = pd.DataFrame()

train = train.drop(['id','digit','letter'],1)
train_a = train_a.drop(['id','letter'],1)

tmp1 = pd.concat([train,train_a])

dftrain = tmp1.to_numpy()

print(dftrain.shape)

dftrain = dftrain.reshape(-1,28,28)


for i in range(dftrain.shape[0]):
    img=dftrain[i]
    img=np.where((img<60)&(img!=0), 0, img)
    img=np.where((img>60)&(img!=0), 254, img)
    img=Image.fromarray(img.astype('uint8'))
    img.save('C:/data/vision2/cut/%d.png'%i)
