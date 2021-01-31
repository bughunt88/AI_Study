# 모델은 RandomForest 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 wine


import numpy as np
from sklearn.datasets import load_wine
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
dataset = load_wine()
x = dataset.data
y = dataset.target

# x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 

kfold = KFold(n_splits=5, shuffle=True)


parameters = [
    {'mal__n_estimators' : [100,200]},
    {'mal__max_depth' : [6,8,10,12]},
    {'mal__min_samples_leaf' : [3,5,7,10]},
    {'mal__n_jobs' : [-1]}
]

# 2. 모델


for train_index, test_index in kfold.split(x) :

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestClassifier())])
    
    score = cross_val_score(model, x_train, y_train, cv=kfold )
    print('교차검증 점수 : ', score)


'''

pipe = Pipeline([ ("scaler", MinMaxScaler()), ('mal', RandomForestClassifier()) ])
#pipe = make_pipeline(n, RandomForestRegressor())

model = GridSearchCV(pipe, parameters, cv=kfold)
# GridSearchCV 최적의 값 찾는 것이 5번

score = cross_val_score(model, x, y, cv=kfold)
# cross_val_score 최적의 점수 찾는 것이 5번 

print('교차검증 점수 : ', score)

'''