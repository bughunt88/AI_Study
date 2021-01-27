from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris

import warnings

warnings.filterwarnings('ignore')


dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) 

# 분류형 모델 전부를 all_estimators에 담았다 
allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, "의 정답률 : ", accuracy_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')

# import sklearn
# print(sklearn.__version__)

'''
AdaBoostClassifier 의 정답률 :  0.9333333333333333
BaggingClassifier 의 정답률 :  0.8888888888888888
BernoulliNB 의 정답률 :  0.28888888888888886     
CalibratedClassifierCV 의 정답률 :  0.8444444444444444
CategoricalNB 의 정답률 :  0.9111111111111111
CheckingClassifier 의 정답률 :  0.3333333333333333
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  0.6222222222222222
DecisionTreeClassifier 의 정답률 :  0.9111111111111111
DummyClassifier 의 정답률 :  0.3333333333333333
ExtraTreeClassifier 의 정답률 :  0.8444444444444444
ExtraTreesClassifier 의 정답률 :  0.9555555555555556
GaussianNB 의 정답률 :  0.9555555555555556
GaussianProcessClassifier 의 정답률 :  0.9333333333333333
GradientBoostingClassifier 의 정답률 :  0.8888888888888888
HistGradientBoostingClassifier 의 정답률 :  0.9111111111111111
KNeighborsClassifier 의 정답률 :  0.9555555555555556
LabelPropagation 의 정답률 :  0.9333333333333333
LabelSpreading 의 정답률 :  0.9333333333333333

LinearDiscriminantAnalysis 의 정답률 :  1.0

LinearSVC 의 정답률 :  0.9555555555555556
LogisticRegression 의 정답률 :  0.9777777777777777
LogisticRegressionCV 의 정답률 :  0.9777777777777777
MLPClassifier 의 정답률 :  0.9333333333333333
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  0.7555555555555555
NearestCentroid 의 정답률 :  0.9111111111111111
NuSVC 의 정답률 :  0.9777777777777777
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  0.8444444444444444
Perceptron 의 정답률 :  0.6222222222222222

QuadraticDiscriminantAnalysis 의 정답률 :  1.0

RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.8222222222222222
RidgeClassifierCV 의 정답률 :  0.8222222222222222
SGDClassifier 의 정답률 :  0.6222222222222222
SVC 의 정답률 :  0.9777777777777777
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''