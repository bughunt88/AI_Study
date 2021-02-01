

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

(x_train, y_train), (x_test,y_test)= mnist.load_data()

# plt.imshow(x_train[1])
# plt.show()


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


#2. 모델
model = XGBClassifier(n_jobs=8,eval_metric='mlogloss')

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ', acc)



# CNN
# 0.02563497982919216
# 0.991599977016449
# y_test :  [7 2 1 0 4 1 4 9 5 9]
# y_predict_argmax :  [7 2 1 0 4 1 4 9 5 9]


# DNN
# 0.13371659815311432
# 0.9804999828338623
# y_test :  [7 2 1 0 4 1 4 9 5 9]
# y_predict_argmax :  [7 2 1 0 4 1 4 9 5 9]


# xgb 
# acc :  0.9625238095238096