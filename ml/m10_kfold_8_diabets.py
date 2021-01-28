# 모델 5개 돌려서 결과치 확인하라 

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import  r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression



dataset = load_diabetes()
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
model_score :  0.4306382618527731
r2_score :  0.4306382618527731
scores :  [0.35215932 0.42396615 0.40857999 0.40676588 0.12758635]

DecisionTreeRegressor()
model_score :  -0.017222806871842744
r2_score :  -0.017222806871842744
scores :  [ 0.09434352 -0.22861312  0.00557439 -0.38802274 -0.48538506]

RandomForestRegressor()
model_score :  0.4871094825662975
r2_score :  0.4871094825662975
scores :  [0.46678072 0.57989198 0.35506727 0.3328724  0.32376228]

LinearRegression()
model_score :  0.4882683148592527
r2_score :  0.4882683148592527
scores :  [0.4712074  0.4009158  0.56500243 0.42029613 0.47202637]
'''