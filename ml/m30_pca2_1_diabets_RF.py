# PCA - 데이터 컬럼을 특성에 맞춰서 줄여주는 방법
# 통상적으로 pca.explained_variance_ratio_의 합이 0.95 이상이면 성능 비슷하다 

# 랜포로 모델링 해보자 

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


datasets = load_diabetes()

x = datasets.data
y = datasets.target


pca = PCA(n_components=9)
x2 = pca.fit_transform(x)


parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'n_jobs' : [-1,2,4]}
]


# 2. 모델

x_train, x_test, y_train, y_test = train_test_split(x2, y,  train_size=0.7, random_state = 77, shuffle=True ) 

kfold = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

# 3. 훈련

model.fit(x_train, y_train)

print('최적의 매개변수 : ', model.best_estimator_)

y_pred = model.predict(x_test)

print('최종정답률 : ', accuracy_score(y_test,y_pred))

#print('모델정답률 : ', model.score(y_test,y_pred))
