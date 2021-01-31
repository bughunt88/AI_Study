
# 최적의 툴을 찾아주는 코드 
# 넣은 파라미터 값을 전부다 사용하는것이 아니라 일부분만 사용해서 찾는 코드

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

from sklearn.pipeline import Pipeline, make_pipeline

import  warnings
warnings.filterwarnings('ignore')


# 1. 데이터 

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# 2. 모델 

# model = Pipeline([ ("scaler", MinMaxScaler()), ('malddong', SVC()) ])
model = make_pipeline(MinMaxScaler(), SVC())

# SVC 라는 모델을 MinMaxScaler랑 합치는 코드 
# 전처리랑 모델을 엮는다 

# 2가지 방법으로 사용할 수 있다

model.fit(x_train, y_train)

results = model.score(x_test, y_test)

print(results)