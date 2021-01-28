
# 모델 5개 돌려서 결과치 확인하라 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 


kfold = KFold(n_splits=5, shuffle=True)

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
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
model_score :  0.9415204678362573
accuracy_score :  0.9415204678362573
scores :  [0.9        0.9125     0.85       0.87341772 0.89873418]

SVC()
model_score :  0.9122807017543859
accuracy_score :  0.9122807017543859
scores :  [0.9625     0.95       0.8875     0.87341772 0.84810127]

KNeighborsClassifier()
model_score :  0.9415204678362573
accuracy_score :  0.9415204678362573
scores :  [0.925      0.95       0.9375     0.89873418 0.92405063]

DecisionTreeClassifier()
model_score :  0.9298245614035088
accuracy_score :  0.9298245614035088
scores :  [0.925      0.925      0.9625     0.89873418 0.94936709]

RandomForestClassifier()
model_score :  0.9532163742690059
accuracy_score :  0.9532163742690059
scores :  [0.925      0.95       0.9625     0.96202532 1.        ]

LogisticRegression()
model_score :  0.9532163742690059
accuracy_score :  0.9532163742690059
scores :  [0.9625     0.95       0.8875     0.93670886 0.97468354]

'''