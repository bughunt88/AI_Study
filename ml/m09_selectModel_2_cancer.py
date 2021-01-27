from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer


import warnings

warnings.filterwarnings('ignore')


dataset = load_breast_cancer()
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


'''
AdaBoostClassifier 의 정답률 :  0.9532163742690059
BaggingClassifier 의 정답률 :  0.9473684210526315
BernoulliNB 의 정답률 :  0.6432748538011696      
CalibratedClassifierCV 의 정답률 :  0.8947368421052632
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 :  0.3567251461988304
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  0.8888888888888888
DecisionTreeClassifier 의 정답률 :  0.9473684210526315
DummyClassifier 의 정답률 :  0.5789473684210527
ExtraTreeClassifier 의 정답률 :  0.9181286549707602

ExtraTreesClassifier 의 정답률 :  0.9766081871345029

GaussianNB 의 정답률 :  0.9473684210526315
GaussianProcessClassifier 의 정답률 :  0.8947368421052632
GradientBoostingClassifier 의 정답률 :  0.9649122807017544
HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
KNeighborsClassifier 의 정답률 :  0.9064327485380117
LabelPropagation 의 정답률 :  0.3684210526315789
LabelSpreading 의 정답률 :  0.3684210526315789
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
LinearSVC 의 정답률 :  0.8947368421052632
LogisticRegression 의 정답률 :  0.9415204678362573
LogisticRegressionCV 의 정답률 :  0.9590643274853801
MLPClassifier 의 정답률 :  0.8947368421052632
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  0.8830409356725146
NearestCentroid 의 정답률 :  0.8713450292397661
NuSVC 의 정답률 :  0.8713450292397661
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  0.9122807017543859
Perceptron 의 정답률 :  0.8304093567251462
QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 :  0.9649122807017544
RidgeClassifier 의 정답률 :  0.9649122807017544
RidgeClassifierCV 의 정답률 :  0.9649122807017544
SGDClassifier 의 정답률 :  0.9005847953216374
SVC 의 정답률 :  0.8888888888888888
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''