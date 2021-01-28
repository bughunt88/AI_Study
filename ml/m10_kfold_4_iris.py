
# 모델 5개 돌려서 결과치 확인하라 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 


kfold = KFold(n_splits=5, shuffle=True)

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for i in models:
    model = i

    #3. compile fit
    #model.fit(x_train,y_train)

    
    score = cross_val_score(model,x_train,y_train, cv=kfold)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print('accuracy_score : ', accuracy)
    print('scores : ', score)
    

'''
LinearSVC()
model_score :  0.8888888888888888
accuracy_score :  0.8888888888888888
scores :  [1.        1.        1.        1.        0.9047619]

SVC()
model_score :  0.9111111111111111
accuracy_score :  0.9111111111111111
scores :  [0.85714286 1.         0.95238095 1.         0.9047619 ]

KNeighborsClassifier()
model_score :  0.9333333333333333
accuracy_score :  0.9333333333333333
scores :  [0.95238095 1.         0.95238095 1.         1.        ]

DecisionTreeClassifier()
model_score :  0.8666666666666667
accuracy_score :  0.8666666666666667
scores :  [1.         1.         0.95238095 0.95238095 1.        ]

RandomForestClassifier()
model_score :  0.8666666666666667
accuracy_score :  0.8666666666666667
scores :  [1.         0.95238095 1.         1.         0.95238095]

'''

