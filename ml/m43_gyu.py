

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import datetime
start1 = datetime.datetime.now()
from xgboost import XGBClassifier,plot_importance, XGBRegressor
# 1. 데이터

x, y = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )


parameters = [
    {'n_estimators' : [100, 200, 300], 'learing_rate' : [0.1,0.3,0.001,0.01],
    'max_depth' : [4,5,6]},
    {'n_estimators' : [90, 100, 110], 'learing_rate' : [0.1,0.001,0.01],
    'max_depth' : [4,5,6], 'colsample_bytree' : [0.6, 0.9, 1]},
]

model = RandomizedSearchCV(XGBRegressor(), parameters, verbose=1)
scores = cross_val_score(model, x_train, y_train)
model.fit(x_train,y_train)
print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률 :', r2_score(y_test, y_pred))
print("scores : ", scores)

thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)
print(np.sum(thresholds))

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True) #thresh값 이상의 것을 전부 처리     디폴트 prefit = False
    select_x_trian = selection.transform(x_train)
    print(select_x_trian.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_trian, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d R2: %.2f%%" %(thresh, select_x_trian.shape[1], score*100))  #부스트 트리 계열에서 사용