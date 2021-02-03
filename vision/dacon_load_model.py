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

#train = pd.read_csv('/content/drive/My Drive/vision/train.csv')
#submission = pd.read_csv('/content/drive/My Drive/vision/sample_submission.csv')


train = pd.read_csv('../data/vision/train.csv')
submission = pd.read_csv('../data/vision/submission.csv')
test = pd.read_csv('../data/vision/test.csv')

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255.
y = temp.iloc[:,1]
x_test = test_df.iloc[:,2:]/255.

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_test.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 , shuffle=True)

model = joblib.load('../data/vision/checkpoint/checkpoint2.dat')

print("로드 완료!")


score = model.score(x_test, y_test)

print("score : ", score)

y_pred = model.predict(x_pred)

pred = pd.DataFrame(y_pred)

submission.loc[:, 'digit'] = pred

submission.to_csv('../data/vision/file/submission.csv', index = False)



print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')