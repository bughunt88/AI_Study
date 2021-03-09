
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

wine = pd.read_csv('../data/csv/winequality-white.csv', sep = ';', header = 0)

wine_npy = wine.values

x = wine_npy[:, :-1]
y = wine_npy[:, -1]


newlist = []
for i in list(y):
    if i <= 4: 
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist



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

