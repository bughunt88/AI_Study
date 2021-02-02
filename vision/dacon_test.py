import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
import joblib
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array # Image Related
import matplotlib.pyplot as plt

#train = pd.read_csv('/content/drive/My Drive/vision/train.csv')
#submission = pd.read_csv('/content/drive/My Drive/vision/sample_submission.csv')


train = pd.read_csv('../data/vision/train.csv')
submission = pd.read_csv('../data/vision/submission.csv')
test = pd.read_csv('../data/vision/test.csv')

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]
y = temp.iloc[:,1]

x = x.to_numpy()
y = y.to_numpy()

print(x.shape)

X_image=x.reshape(-1,28,28,1)

# 이미지 생성기의 선언
datagen = ImageDataGenerator(
                                 width_shift_range=5,
                                 height_shift_range=5,
                                 rotation_range=10,
                                 zoom_range=0.05)  


# flow형태의 정의
flow1=datagen.flow(X_image,y,batch_size=32,seed=2020) 


print(flow1[0])

X_image_gen1,X_letter_gen=flow1.next()

## 생성된 데이터의 형태 확인
print("X_image_gen1.shape={}".format(X_image_gen1.shape))
print("letter_gen.shape={}".format(X_letter_gen.shape))

