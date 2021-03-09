# 실습


# 케라스로 to_categorical하면 최대, 최소 상관 없이 무조건 0부터 쓴다 !
# OneHotEncoder은 최소값부터 값을 잡는다 !
# sparse_categorical_crossentropy은 확인해보자 


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('../data/naver/winequality-white.csv', sep=';')

df= df.values

x = df[:,:-1]
y = df[:,-1]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=66, shuffle=True, train_size=0.8)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

model = RandomForestClassifier()

model.fit(x_train, y_train)

score = model.score(x_test,y_test)

print("score : ",score)

