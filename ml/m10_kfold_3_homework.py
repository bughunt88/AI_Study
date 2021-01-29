
# trina, test 나눈 다음에 train만 발리데이션 하지 말고, 
# kfold 한 후에 train_test_split 사용 

# 5등분 후 트레인 테스트 적용 

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

kfold = KFold(n_splits=5, shuffle=True)

model = LinearSVC()

model.fit(x,y)


for train_index, test_index in kfold.split(x): 
    #print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.7, random_state = 77, shuffle=True ) 



score = cross_val_score(model,x_train,y_train, cv=kfold)


print('scores : ', score)


