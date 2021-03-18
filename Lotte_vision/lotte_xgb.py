
# XGB에 수정했을 시 유의미한 파라미터들 
'''
parameters = [
    {"n_estimators" : [100,200,300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]},
    {"n_estimators" : [90,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd


import  warnings
warnings.filterwarnings('ignore')

sub = pd.read_csv('../data/LPD_competition/sample.csv', header = 0)


x_train = np.load('../data/lpd_competition/npy/train_data_x9.npy')
y_train = np.load('../data/lpd_competition/npy/train_data_y9.npy')
x_test = np.load('../data/lpd_competition/npy/predict_data9.npy')

x_train = x_train.reshape(60000, x_train.shape[1]*x_train.shape[3]*x_train.shape[4]).astype('float32')/255.
x_test = x_test.reshape(60000, x_test.shape[1]*x_test.shape[3]*x_test.shape[4]).astype('float32')/255.

print(x_train.shape)

kfold = KFold(n_splits=5, shuffle=True)


# XGB에 수정했을 시 유의미한 파라미터들 
parameters = [
    {"n_estimators" : [100,200,300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]},
    {"n_estimators" : [90,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]


# 2. 모델

model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold)
# 파라미터 값들(dict 형태)을 모두 돌려주는 코드
# 하나의 모델로 볼 수 있다


# 3. 훈련

model.fit(x_train, y_train)


# 3. 평가, 예측
# score = cross_val_score(model,x_train,y_train, cv=kfold)

print('최적의 매개변수 : ', model.best_estimator_)
# model.best_estimator_ : 위에 파라미터 값을 넣은 것 중 최고를 뽑아서 알려준다 

y_pred = model.predict(x_test)

sub.loc[:,'prediction'] = y_pred
sub.to_csv('../data/LPD_competition/sample_011.csv', index = False)

#print('최종정답률 : ', accuracy_score(y_test,y_pred))


#print('모델정답률 : ', model.score(y_test,y_pred))

