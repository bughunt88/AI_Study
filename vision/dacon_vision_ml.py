import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
import joblib

#train = pd.read_csv('/content/drive/My Drive/vision/train.csv')
#submission = pd.read_csv('/content/drive/My Drive/vision/sample_submission.csv')


train = pd.read_csv('../data/vision/train.csv')
submission = pd.read_csv('../data/vision/submission.csv')
test = pd.read_csv('../data/vision/test.csv')

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255.
y = temp.iloc[:,1]
x_test = test_df.iloc[:,2:]/255.

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_test.to_numpy()

'''
pca = PCA(n_components=400)
x = pca.fit_transform(x)
'''


# x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 , shuffle=True)


# 2. 모델

kfold = KFold(n_splits=5, shuffle=True)

model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.2,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=None, n_estimators=600, n_jobs=8,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

acc_list = []
for train_index, test_index in kfold.split(x): 
    #print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #3. 훈련
    model.fit(x_train,y_train,verbose=True, eval_set=[(x_train,y_train),(x_test,y_test)])

    #4. 평가 예측
    acc = model.score(x_test,y_test)

    acc_list.append(acc)

    joblib.dump(model, '../data/vision/checkpoint/checkpoint'+str(len(acc_list))+'.dat')



print(acc_list)


'''

y_pred = model.predict(x_pred)

pred = pd.DataFrame(y_pred)

submission.loc[:, 'digit'] = pred

submission.to_csv('../data/vision/file/test1.csv', index = False)

'''

print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')