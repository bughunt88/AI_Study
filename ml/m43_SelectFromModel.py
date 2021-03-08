from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

model = XGBRegressor(n_jobs = 8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 : ', score)

thresholds = np.sort(model.feature_importances_) # 이렇게 하면 fi 값들이 정렬되어 나온다!
print(thresholds)

# 각각의 피쳐마다 중요도가 있는데, 중요도 낮은순으로 하나씩 빼면서 포문을 돌린거다!!
# 결과물 출력시 오히려 4개의 피쳐를 뺀 결과가 제일 좋았다

# 과제1. prefit 에 대해서 알아보기


# 하나씩 포문돌린다
for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)

    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))

#print(model.coef_)
#print(model.intercept_)