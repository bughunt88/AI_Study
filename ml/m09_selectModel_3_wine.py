from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine


import warnings

warnings.filterwarnings('ignore')


dataset = load_wine()
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
AdaBoostClassifier 의 정답률 :  0.5370370370370371
BaggingClassifier 의 정답률 :  0.9814814814814815
BernoulliNB 의 정답률 :  0.4074074074074074      
CalibratedClassifierCV 의 정답률 :  0.9444444444444444
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 :  0.4074074074074074
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  0.7407407407407407
DecisionTreeClassifier 의 정답률 :  0.9814814814814815
DummyClassifier 의 정답률 :  0.2962962962962963
ExtraTreeClassifier 의 정답률 :  0.8333333333333334

ExtraTreesClassifier 의 정답률 :  1.0

GaussianNB 의 정답률 :  0.9814814814814815
GaussianProcessClassifier 의 정답률 :  0.37037037037037035
GradientBoostingClassifier 의 정답률 :  0.9629629629629629

HistGradientBoostingClassifier 의 정답률 :  1.0

KNeighborsClassifier 의 정답률 :  0.6851851851851852
LabelPropagation 의 정답률 :  0.5185185185185185
LabelSpreading 의 정답률 :  0.5185185185185185
LinearDiscriminantAnalysis 의 정답률 :  0.9814814814814815
LinearSVC 의 정답률 :  0.8703703703703703
LogisticRegression 의 정답률 :  0.9629629629629629
LogisticRegressionCV 의 정답률 :  0.9629629629629629
MLPClassifier 의 정답률 :  0.7777777777777778
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  0.8333333333333334
NearestCentroid 의 정답률 :  0.6851851851851852
NuSVC 의 정답률 :  0.9444444444444444
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  0.6851851851851852
Perceptron 의 정답률 :  0.7407407407407407
QuadraticDiscriminantAnalysis 의 정답률 :  0.9629629629629629
RadiusNeighborsClassifier 은 없는 놈!

RandomForestClassifier 의 정답률 :  1.0

RidgeClassifier 의 정답률 :  0.9814814814814815
RidgeClassifierCV 의 정답률 :  0.9814814814814815
SGDClassifier 의 정답률 :  0.42592592592592593
SVC 의 정답률 :  0.6666666666666666
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''