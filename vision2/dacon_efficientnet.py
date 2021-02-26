
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

model = efn.EfficientNetB0(weights='imagenet')


#### 이미지 전처리 

image = imread('../data/vision2/cat.jpg')
#image = imread('../data/vision2/test_dirty_mnist_2nd/50001.png')

image_size = model.input_shape[1] # 224

x = efn.center_crop_and_resize(image, image_size=image_size)

x = efn.preprocess_input(x)

x = np.expand_dims(x, 0)

x = x.reshape(x.shape[0],x.shape[1],x.shape[2],3)

####

y = model.predict(x)

dy = decode_predictions(y)

print(dy)