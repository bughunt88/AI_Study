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

'''
alphabets = string.ascii_lowercase
alphabets = list(alphabets)
'''

# train 데이터

train = pd.read_csv('../data/vision2/mnist_data/train.csv')



# 256, 256 이미지를 돌리면 터진다 
# 안 터지도록 수정을 해야함 
# test 데이터가 50000만개가 필요할까??



# train 데이터 
# 1회에 쓰던 mnist 데이터 A를 모아서 256으로 리사이징 해준다 
# 알파벳 별로 모델을 만들 것!

train2 = train.drop(['id','digit'],1)

train2['y_train'] = 1

a_train = train2.loc[train2['letter']=='A']
a_train = a_train.drop(['letter'],1)

x_train = a_train.to_numpy().astype('int32')[:,:-1] # (72, 784)
y_train = a_train.to_numpy()[:,-1] # (72, 1)

# 이미지 전처리 100보다 큰 것은 253으로 변환, 100보다 작으면 0으로 변환
x_train[100 < x_train] = 253
x_train[x_train < 100] = 0

x_train = x_train.reshape(-1,28,28,1)
x_train = experimental.preprocessing.Resizing(256,256)(x_train)


np.save('../data/vision2/x_data_save.npy', arr=x_train)
np.save('../data/vision2/y_data_save.npy', arr=y_train)

