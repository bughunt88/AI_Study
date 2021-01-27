from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston

import warnings

warnings.filterwarnings('ignore')


dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) 

# 회기 모델 전부를 all_estimators에 담았다 
allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, "의 정답률 : ", r2_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')

'''
ARDRegression 의 정답률 :  0.783438698567686
AdaBoostRegressor 의 정답률 :  0.8596033642182015
BaggingRegressor 의 정답률 :  0.8626835288580957
BayesianRidge 의 정답률 :  0.7912228365229351
CCA 의 정답률 :  0.7757272685646832
DecisionTreeRegressor 의 정답률 :  0.6416991660838834
DummyRegressor 의 정답률 :  -0.005227869326375867
ElasticNet 의 정답률 :  0.7364371198416895
ElasticNetCV 의 정답률 :  0.7213117448627095
ExtraTreeRegressor 의 정답률 :  0.5880926401206807
ExtraTreesRegressor 의 정답률 :  0.8952547824443619
GammaRegressor 의 정답률 :  -0.005227869326375867
GaussianProcessRegressor 의 정답률 :  -5.81311213590078
GeneralizedLinearRegressor 의 정답률 :  0.7329801918837436

GradientBoostingRegressor 의 정답률 :  0.9120924453596585

HistGradientBoostingRegressor 의 정답률 :  0.8962100389153277
HuberRegressor 의 정답률 :  0.7491371356709582
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 :  0.6338244105803171
KernelRidge 의 정답률 :  0.787165248336516
Lars 의 정답률 :  0.8044888426543628
LarsCV 의 정답률 :  0.8032830033921297
Lasso 의 정답률 :  0.7234368838497398
LassoCV 의 정답률 :  0.757958048006295
LassoLars 의 정답률 :  -0.005227869326375867
LassoLarsCV 의 정답률 :  0.8044516427844497
LassoLarsIC 의 정답률 :  0.7983441148086405
LinearRegression 의 정답률 :  0.8044888426543619
LinearSVR 의 정답률 :  0.5421663266245169
MLPRegressor 의 정답률 :  0.5751678725511221
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 :  0.2886896021336892
OrthogonalMatchingPursuit 의 정답률 :  0.5651272222459414
OrthogonalMatchingPursuitCV 의 정답률 :  0.7415292549226284
PLSCanonical 의 정답률 :  -2.2717245026237833
PLSRegression 의 정답률 :  0.7738717095948149
PassiveAggressiveRegressor 의 정답률 :  0.213631628346159
PoissonRegressor 의 정답률 :  0.8524929869037771
RANSACRegressor 의 정답률 :  -7.192695601818386
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 :  0.8849603443186239
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  0.8031981173042202
RidgeCV 의 정답률 :  0.8046761789224935
SGDRegressor 의 정답률 :  -4.3636261483428926e+26
SVR 의 정답률 :  0.26986966510824173
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  0.7664722308023031
TransformedTargetRegressor 의 정답률 :  0.8044888426543619
TweedieRegressor 의 정답률 :  0.7329801918837436
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''
