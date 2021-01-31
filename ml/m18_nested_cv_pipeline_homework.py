# 모델은 RandomForest 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 diabets

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline, make_pipeline


import  warnings
warnings.filterwarnings('ignore')


# 1. 데이터 

dataset = load_diabetes()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)

# kfold로 5라인 x_train, x_test 만들기 


# 2. 모델

# kfold로 나누어진 데이터를 for 돌려서 train만 파이프 라인에 넣는다 
for train_index, test_index in kfold.split(x) :

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestClassifier())])

    score = cross_val_score(model, x_train, y_train, cv=kfold )
    print('교차검증 점수 : ', score)
