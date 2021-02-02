from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 66)


# 2. 모델
model = XGBClassifier(n_estimators=10, learing_rate = 0.01, n_jobs = 8)


# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['error','logloss'], eval_set=[(x_train, y_train), (x_test, y_test)])

aaa = model.score(x_test, y_test)

print("aaa : ", aaa)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("acc : ", acc)

print("##########################")
results = model.evals_result()
print(results)



