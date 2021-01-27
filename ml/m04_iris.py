
# 분류 머신 러닝 다양한 모델 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



dataset = load_iris()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) 

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print('accuracy_score : ', accuracy)
    

# TensorFlow
# loss:  [0.10291372239589691, 0.9666666388511658]
# y_predict_argmax:  [2 0 0 2]


# LinearSVC
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 1 1 0 1 2 1 1] 의 예측결과 :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 2 2 0 0 1 2 0 0 2 0 1 0 1
# 0 2 1 0 2 2 1 2]
# model.score :  0.8888888888888888
# accuracy_score :  0.8888888888888888

# SVC
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 1 1 0 1 2 1 1] 의 예측결과 :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 2 1 0 1 2 1 1]
# model.score :  0.9777777777777777
# accuracy_score :  0.9777777777777777

# KNeighborsClassifier
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 1 1 0 1 2 1 1] 의 예측결과 :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 2 1 0 1 2 1 1]
# model.score :  0.9777777777777777
# accuracy_score :  0.9777777777777777

# DecisionTreeClassifier
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 1 1 0 1 2 1 1] 의 예측결과 :  [1 1 1 0 1 1 0 0 0 1 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 2 1 0 2 2 1 1]
# model.score :  0.9111111111111111
# accuracy_score :  0.9111111111111111

# RandomForestClassifier
# [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2 0 0 2 0 1 0 1
# 0 1 1 0 1 2 1 1] 의 예측결과 :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 0 1 2 0 0 2 0 1 0 1
# 0 2 1 0 2 2 1 1]
# model.score :  0.9111111111111111
# accuracy_score :  0.9111111111111111
