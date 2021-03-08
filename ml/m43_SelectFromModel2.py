# 실습
# 1. 상단 모델에 그리드 서치 또는 랜덤서치로 듀닝한 모델 구성
# 최적의 R2 값과 피처임포턴스 구할 것 

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서 
# 최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처를)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용하여
# 최적의 R2 구할 것

# 1번값과 2번값 비교

# XGB에 수정했을 시 유의미한 파라미터들 
'''
parameters = [
    {"n_estimators" : [100,200,300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]},
    {"n_estimators" : [90,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]
'''

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier,XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score


import  warnings
warnings.filterwarnings('ignore')

# 1. 데이터 
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators" : [100,200,300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]},
    {"n_estimators" : [90,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]

model = GridSearchCV(XGBRegressor(), parameters, cv=kfold)
model.fit(x_train, y_train)
print('최적의 매개변수 : ', model.best_estimator_)

print("###################################")

thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)

# 하나씩 포문돌린다
tmp = 0
tmp2 = [0,0]

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=110, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
    
    # selection_model = model.best_estimator_

    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    if score > tmp :
        tmp = score
        tmp2[0] = thresh
        tmp2[1] = select_x_train.shape[1]

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))


selection = SelectFromModel(model.best_estimator_, threshold = tmp2[0], prefit = True)

select_x_train = selection.transform(x_train)

selection_model = GridSearchCV(XGBRegressor(), parameters, cv =5)
selection_model.fit(select_x_train, y_train)

select_x_test = selection.transform(x_test)
y_predict = selection_model.predict(select_x_test)

score = r2_score(y_test, y_predict)

print("최종 스코어 : ", score*100)