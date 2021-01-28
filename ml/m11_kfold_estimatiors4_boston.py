from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston

import warnings

warnings.filterwarnings('ignore')


dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) 


# 데이터를 원하는 수로 나누기만 한다 
kfold = KFold(n_splits=5, shuffle=True)


# 분류형 모델 전부를 all_estimators에 담았다 
allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        #model.fit(x_train, y_train)
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        #y_pred = model.predict(x_test)
        print(name, "의 정답률 : \n", score)
        print("\n")
    except:
        # continue
        print(name, '은 없는 놈!')
        print("\n")
# import sklearn
# print(sklearn.__version__)
