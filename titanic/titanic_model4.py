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

y_train = main_data[:,0]
x_train = main_data[:,1:]

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

kfold = KFold(n_splits=24, shuffle=True)

num = 0 

rmse_list = []
loss_list = []

for train_index, test_index in kfold.split(x_train): 

    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]

    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=True ) 

    scaler = MinMaxScaler()
    scaler.fit(x_train1)
    x_train1 = scaler.transform(x_train1)
    x_test1 = scaler.transform(x_test1)
    x_val = scaler.transform(x_val)
    x_pred = scaler.transform(x_pred)

    # 2. 모델구성

    model = Sequential()
    model.add(Dense(2048, activation='swish' ,input_dim= 6))
    model.add(Dropout(0.2))
    model.add(Dense(2048,activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(2048,activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(2048,activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='swish'))
    model.add(Dense(16,activation='swish'))
    model.add(Dense(1, activation='sigmoid')) 

    # 3. 컴파일 훈련

    modelpath = '../data/modelcheckpoint/titanic_2_'+str(num)+'.hdf5'

    es= EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, verbose=1)
    cp =ModelCheckpoint(filepath=modelpath, save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train1, y_train1, epochs=1000, batch_size=32, validation_data=(x_val,y_val), callbacks=[es,reduce_lr,cp] )

    # 4. 평가, 예측

    loss, mae = model.evaluate(x_test1, y_test1, batch_size=32)
    y_predict = model.predict(x_pred)

    rmse_list.append(mae)
    loss_list.append(loss)

    num += 1


print("k-fold 확인")
print("mae : ",rmse_list)
print("loss : ",loss_list)
