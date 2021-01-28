
# 최적의 툴을 찾아주는 코드 
# 넣은 파라미터 값을 전부다 사용하는것이 아니라 일부분만 사용해서 찾는 코드
# 속도가 빠르고 GridSearchCV랑 성능도 비슷하다

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


import  warnings
warnings.filterwarnings('ignore')

# 1. 데이터 

import pandas as pd

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)

x = dataset.iloc[:,:-1] 
y = dataset.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 

kfold = KFold(n_splits=5, shuffle=True)


parameters = [
    {"C" : [1,10,100,1000], "kernel":["linear"]},
    {"C" : [1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},
    {"C" : [1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001,0.0001]}
]


# 2. 모델

model = RandomizedSearchCV(SVC(), parameters, cv=kfold)
# 파라미터 값들(dict 형태)을 모두 돌려주는 코드
# 하나의 모델로 볼 수 있다


# 3. 훈련

model.fit(x_train, y_train)


# 3. 평가, 예측
# score = cross_val_score(model,x_train,y_train, cv=kfold)

print('최적의 매개변수 : ', model.best_estimator_)
# model.best_estimator_ : 위에 파라미터 값을 넣은 것 중 최고를 뽑아서 알려준다 

y_pred = model.predict(x_test)

print('최종정답률 : ', accuracy_score(y_test,y_pred))
