import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

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

scaler = MinMaxScaler()
scaler.fit(x_pred)
x_pred = scaler.transform(x_pred)


from tensorflow.keras.models import load_model

result = 0


check = 24


for main_num in range(check): 

    filepath_cp = f'../data/modelcheckpoint/titanic_2_'+str(main_num)+'.hdf5'
    model = load_model(filepath_cp, compile = False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_predict = model.predict(x_pred)

    result += y_predict

result = result / check

y_pred_r = np.where(result<0.5,0,1)

sub = pd.read_csv('../data/titanic/sample_submission.csv')
sub['Survived'] = y_pred_r
sub.to_csv('../data/titanic/sample_003.csv',index=False)

