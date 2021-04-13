import numpy as np
import db_connect as db
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


query = "SELECT * FROM main_data_table"


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()



train_value = df[ '2020-05-01' >= df['date'] ]

x_train = train_value.iloc[:100,1:-1]
y_train = train_value.iloc[:100,-1]

test_value = df[df['date'] >=  '2020-05-01']

x_test = test_value.iloc[:100,1:-1]
y_test = test_value.iloc[100:,-1]
x_test = x_test.dropna()    # 결측값 제거
x_train = x_train.dropna()    # 결측값 제거
y_test = y_test.dropna()    # 결측값 제거
y_train = y_train.dropna()    # 결측값 제거

# print(x_test.head())
print(y_test.head())

x_train = train_value.iloc[:,1:-1].to_numpy()
y_train = train_value.iloc[:,-1].to_numpy()
x_test = test_value.iloc[:,1:-1].to_numpy()
y_test = test_value.iloc[:,-1].to_numpy()

# StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.8, random_state = 77, shuffle=True ) 

print(x_test)
print(x_test.shape)

# 2. 모델구성

model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 6)) 
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
# model.add(Dense(64))
# model.add(Dense(64))
model.add(Dense(1)) 
model.summary()
# 3. 컴파일 훈련

# modelpath = '../data/modelCheckpoint/team1.hdf5'
# es= EarlyStopping(monitor='val_loss', patience=10)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_val,y_val))