# 모델 5개 돌려서 결과치 확인하라 

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import  r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression



dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 


kfold = KFold(n_splits=5, shuffle=True)



models = [KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), LinearRegression()]
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
    r2 = r2_score(y_test,y_pred)
    print('r2_score : ', r2)
    print('scores : ', score)


'''
KNeighborsRegressor()
model_score :  0.63989391830562
r2_score :  0.63989391830562
scores :  [0.29141467 0.39709649 0.4278381  0.17255384 0.53553178]

DecisionTreeRegressor()
model_score :  0.7858280496061781
r2_score :  0.7858280496061781
scores :  [0.73951518 0.69756806 0.71973327 0.79679306 0.81214638]

RandomForestRegressor()
model_score :  0.8869005498574908
r2_score :  0.8869005498574908
scores :  [0.93462655 0.71645487 0.86087777 0.87948053 0.88030334]

LinearRegression()
model_score :  0.7175720869047586
r2_score :  0.7175720869047586
scores :  [0.70604593 0.60408835 0.76567811 0.71705354 0.72266587]
'''