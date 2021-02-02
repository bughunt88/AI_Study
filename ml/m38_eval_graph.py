from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

''' 
# eval_metric

eval_metric: 설정 한 objective기본 설정 값이 지정되어 있습니다.
rmse : 제곱 평균 제곱근 오차
mae : 절대 오류를 의미
logloss : 음의 로그 우도
오류 : 이진 분류 오류율 (0.5 임계 값)
merror : 다중 클래스 분류 오류율
mlogloss : 다중 클래스 logloss
auc : 곡선 아래 영역 
'''

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, shuffle=True, random_state = 1
)

# model
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

# fit
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse','logloss','mae'],
        eval_set=[(x_train,y_train),(x_test,y_test)],
        early_stopping_rounds=10
            )

aaa = model.score(x_test,y_test)
print("score :", aaa)

y_prad = model.predict(x_test)
r2 = r2_score(y_test, y_prad)
print("r2 : ", r2)

print("================================================")

result = model.evals_result()
# 발리데이션 로스 값을 출력
print("result :", result)

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label='Train')
ax.plot(x_axis, result['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost Rmse')
plt.show()
"""
score : 0.9159675084902162
r2 :  0.9159675084902162
"""