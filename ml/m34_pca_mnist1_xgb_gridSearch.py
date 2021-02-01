# m31로 만든 0.95 이상의 n_component = ? 를 사용하여 xgb 모델을 만들 것

# mnist dnn 보다 성능 좋게 만들어라!
# cnn과 비교 
# RandomSearch 로도 해볼 것 


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test,y_test)= mnist.load_data()

x = np.append(x_train, x_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])


pca1 = PCA()
pca1.fit(x)
cumsum = np.cumsum(pca1.explained_variance_ratio_)

pca_num = np.argmax(cumsum >= 0.95)+1

pca = PCA(n_components=pca_num)
x2 = pca.fit_transform(x)


# 데이터 전처리


x_train, x_test, y_train, y_test = train_test_split(x2, y,  train_size=0.7, random_state = 77, shuffle=True ) 

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

kfold = KFold(n_splits=5, shuffle=True)


parameters = [
    {"n_estimators" : [100,200,300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]}
    #,{"n_estimators" : [90,100,110], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]},
    #{"n_estimators" : [90,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]}
]

# 2. 모델


model_list = [GridSearchCV, RandomizedSearchCV]

for n in model_list:

    model = n(XGBClassifier(n_jobs = 8, n_estimators=2000), parameters, cv=kfold)
    # n_estimators - 딥러닝에 epochs 같다 

    # 3. 훈련

    model.fit(x_train, y_train, verbose=True, eval_metric='mlogloss', eval_set=[(x_train, y_train), (x_test, y_test)])
    # eval_metric - 딥러닝에 metric와 같다 
    # eval _set - 딥러닝에 validation과 같다 
    # verbose는 eval_set이 있어야 작동한다 

    # 3. 평가, 예측

    print(n," 모델")

    print('최적의 매개변수 : ', model.best_estimator_)

    y_pred = model.predict(x_test)

    print('최종정답률 : ', accuracy_score(y_test,y_pred))
