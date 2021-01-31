
# 분류 머신 러닝 다양한 모델 

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

score = cross_val_score(model,x,y, cv=kfold)

print('scores : ', score)
