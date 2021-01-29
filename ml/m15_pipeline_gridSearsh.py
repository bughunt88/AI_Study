
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline, make_pipeline

import  warnings
warnings.filterwarnings('ignore')


# 1. 데이터 

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 


# make_pipeline 사용시 파라미터 
'''
parameters = [
    {"svc__C" : [1,10,100,1000], "svc__kernel":["linear"]},
    {"svc__C" : [1,10,100], "svc__kernel":["rbf"], "svc__gamma":[0.001,0.0001]},
    {"svc__C" : [1,10,100,1000], "svc__kernel":["sigmoid"], "svc__gamma":[0.001,0.0001]}
]
'''
# 파라미터 값 앞에 모델 명 + "__" 써서 파라미터 명 정해야 한다 



# Pipeline 사용시 파라미터 

parameters = [
    {"mal__C" : [1,10,100,1000], "mal__kernel":["linear"]},
    {"mal__C" : [1,10,100], "mal__kernel":["rbf"], "mal__gamma":[0.001,0.0001]},
    {"mal__C" : [1,10,100,1000], "mal__kernel":["sigmoid"], "mal__gamma":[0.001,0.0001]}
]

# 파라미터 값 앞에 모델 앞에 지정한 명칭을 적어줘야 함!



# 2. 모델 

models = [MinMaxScaler(), StandardScaler()]

for i in models:

    pipe = Pipeline([ ("scaler", i), ('mal', SVC()) ])
    #pipe = make_pipeline(i, SVC())

    # 모델과 데이터 전처리를 엮는다

    model = GridSearchCV(pipe, parameters, cv=5)
    # 여러번 돌린 것에서 

    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    
    print(i,"결과 : ",results)



