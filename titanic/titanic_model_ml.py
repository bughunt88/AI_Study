import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

'''
0 survival	survival	0 = 아니오, 1 = 예
1 pclass	티켓 등급	1 = 1, 2 = 2, 3 = 3
2 섹스	섹스	
3 Age	나이	
4 sibsp	타이타닉 호에 탑승 한 형제 자매 / 배우자 수	
5 parch	타이타닉 호에 탑승 한 부모 / 자녀 수	
6 ticket	티켓 번호	
7 fare	여객 운임 가격 
8 cabin	객실 번호	
9 embarked	승선 항	C = Cherbourg, Q = 퀸스 타운, S = 사우 샘프 턴
'''

# 수정사항 
# age, fare nan 데이터가 있음
# pclass 기준으로 nan 채움 
# age 같은 경우 회기 모델로 채우면 더 좋을 듯

#### 트레인 데이터 ####

t_data = pd.read_csv('../data/titanic/train.csv', index_col=0,header=0,encoding='CP949')

t_main_data = t_data.iloc[:,[0,1,3,4,5,6,8]]

t_main_data['Sex'] = np.where(t_data['Sex'] != 'male', 0, 1)
# 남자 0, 여자 1

t_main_data["Fare"].fillna(t_main_data.groupby(["Pclass"])["Fare"].transform("median"), inplace=True)
# 티켓 등급 기준으로 요금 평균

# 이 부분은 수정해야한다!!!!!
t_main_data["Age"].fillna(t_main_data.groupby(["Pclass"])["Age"].transform("median"), inplace=True)
# 요금 기준으로 나이 평균

t_main_data = t_main_data.astype({"Age": "int64"})

main_data =  t_main_data.to_numpy()

y_train1 = main_data[:,0]
x_train1 = main_data[:,1:]

###### 프레딕 데이터 #######

p_data = pd.read_csv('../data/titanic/test.csv', index_col=0,header=0,encoding='CP949')
#t_data.replace(',','',inplace=True, regex=True)

p_main_data = p_data.iloc[:,[0,2,3,4,5,7]]
p_main_data['Sex'] = np.where(p_main_data['Sex'] != 'male', 0, 1)

p_main_data["Fare"].fillna(p_main_data.groupby(["Pclass"])["Fare"].transform("median"), inplace=True)
# 티켓 등급 기준으로 요금 평균

# 이 부분은 수정해야한다!!!!!
p_main_data["Age"].fillna(p_main_data.groupby(["Pclass"])["Age"].transform("median"), inplace=True)
# 요금 기준으로 나이 평균

p_main_data = p_main_data.astype({"Age": "int64"})

x_pred =  p_main_data.to_numpy()

################

kfold = KFold(n_splits=3, shuffle=True)

num = 0 

rmse_list = []
loss_list = []

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import catboost as ctb

N_ESTIMATORS = 1000
N_SPLITS = 10
SEED = 2021
EARLY_STOPPING_ROUNDS = 20
VERBOSE = 0

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': SEED,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': N_ESTIMATORS,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

y_pred = 0

#KFold
for fold, (train_idx, valid_idx) in enumerate(skf.split(x_train1, y_train1)) :
    print(f"=====Fold {fold}=====")

    x_train, x_val = x_train1[train_idx], x_train1[valid_idx]
    y_train, y_val = y_train1[train_idx], y_train1[valid_idx]
    # print(x_train.shape, x_val.shape)  # (90000, 20) (10000, 20)
    # print(y_train.shape, y_val.shape)  # (90000,) (10000,)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_pred = scaler.transform(x_pred)

    model = ctb.CatBoostClassifier(**params)
    model.fit(x_train, y_train,
            eval_set=[(x_val, y_val)],
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=VERBOSE)

    y_val_pred = model.predict(x_val)
    acc_score = accuracy_score(y_val, y_val_pred)
    print(f"===== ACCURACY SCORE {acc_score:.6f} =====")    # 0.778700

    y_pred += model.predict(x_pred)


y_pred /= N_SPLITS
y_pred.shape


submission = pd.read_csv('../data/titanic/sample_submission.csv')

submission['Survived'] = np.round(y_pred).astype(int)

submission.to_csv('../data/titanic/sample_007.csv',index=False)
