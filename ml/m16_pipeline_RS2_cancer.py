
# RandomSearch, GS와 Pipeline을 엮어라!!
# 모델은 RandomForest


import numpy as np
from sklearn.datasets import load_breast_cancer
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

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 

# Pipeline 파라미터 

parameters = [
    {'mal__n_estimators' : [100,200]},
    {'mal__max_depth' : [6,8,10,12]},
    {'mal__min_samples_leaf' : [3,5,7,10]},
    {'mal__n_jobs' : [-1,2,4]}
]


# make_pipeline 파라미터 
'''
parameters = [
    {'randomforestclassifier__n_estimators' : [100,200]},
    {'randomforestclassifier__max_depth' : [6,8,10,12]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__n_jobs' : [-1,2,4]}
]
'''

# 2. 모델 

scaler = [MinMaxScaler(), StandardScaler()]
models = [RandomizedSearchCV, GridSearchCV]

for i in models:
    for n in scaler:
        pipe = Pipeline([ ("scaler", n), ('mal', RandomForestClassifier()) ])
        #pipe = make_pipeline(n, RandomForestClassifier())

        model = i(pipe, parameters, cv=5)

        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        
        print(i.__name__,"-",n," 결과 : ",results)

