
# 최적의 툴을 찾아주는 코드 
# 넣은 파라미터 값을 전부다 사용하는것이 아니라 일부분만 사용해서 찾는 코드

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

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 

# 2. 모델 

# model = Pipeline([ ("scaler", MinMaxScaler()), ('malddong', RandomForestRegressor()) ])

models = [MinMaxScaler(), StandardScaler()]
for i in models:

    model = make_pipeline(i, RandomForestRegressor())
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    
    print(i,"결과 : ",results)

