from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes

import warnings

warnings.filterwarnings('ignore')


dataset = load_diabetes()
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
ARDRegression 의 정답률 :  0.5046292259802914
AdaBoostRegressor 의 정답률 :  0.4195077186238706
BaggingRegressor 의 정답률 :  0.32655902883448196       
BayesianRidge 의 정답률 :  0.5144086777149999
CCA 의 정답률 :  0.4723221227632096
DecisionTreeRegressor 의 정답률 :  -0.015280173982129552
DummyRegressor 의 정답률 :  -0.0003678139434499794      
ElasticNet 의 정답률 :  0.008137786055077423
ElasticNetCV 의 정답률 :  0.4459455039820015
ExtraTreeRegressor 의 정답률 :  -0.1242479789766755
ExtraTreesRegressor 의 정답률 :  0.4058265856742034
GammaRegressor 의 정답률 :  0.0059946505184177434
GaussianProcessRegressor 의 정답률 :  -11.911222185492552   
GeneralizedLinearRegressor 의 정답률 :  0.005938054431265827
GradientBoostingRegressor 의 정답률 :  0.39684147320847485
HistGradientBoostingRegressor 의 정답률 :  0.38707520751705093
HuberRegressor 의 정답률 :  0.5162215138923945
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 :  0.38706446604824185
KernelRidge 의 정답률 :  -3.292961498188917
Lars 의 정답률 :  0.311242465413163
LarsCV 의 정답률 :  0.5026578508880556
Lasso 의 정답률 :  0.35186794366719465
LassoCV 의 정답률 :  0.50549943409123
LassoLars 의 정답률 :  0.3875772452195182
LassoLarsCV 의 정답률 :  0.5017719206510016
LassoLarsIC 의 정답률 :  0.511840115265207

LinearRegression 의 정답률 :  0.5209563551242161

LinearSVR 의 정답률 :  -0.35166512464659694
MLPRegressor 의 정답률 :  -2.837661888422559
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 :  0.13634860898199508
OrthogonalMatchingPursuit 의 정답률 :  0.3332835124181348
OrthogonalMatchingPursuitCV 의 정답률 :  0.45845818813944883
PLSCanonical 의 정답률 :  -1.2159770341554732
PLSRegression 의 정답률 :  0.4958323974723071
PassiveAggressiveRegressor 의 정답률 :  0.45505117390177086
PoissonRegressor 의 정답률 :  0.34830763113939334
RANSACRegressor 의 정답률 :  0.3364074885168653
RadiusNeighborsRegressor 의 정답률 :  -0.0003678139434499794
RandomForestRegressor 의 정답률 :  0.4201244600832127
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  0.4136878337244285
RidgeCV 의 정답률 :  0.5085161326812712
SGDRegressor 의 정답률 :  0.39812082079875966
SVR 의 정답률 :  0.1433892240675827
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  0.5101236687064709

TransformedTargetRegressor 의 정답률 :  0.5209563551242161

TweedieRegressor 의 정답률 :  0.005938054431265827
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''