import numpy as np
import db_connect as db
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

query = "select * from main_data_table"


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()


size = 59136 #30
total_data = split_x(x_train,size)



test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

kfold = KFold(n_splits=3, shuffle=True)

num = 0 
r2_list = []
rmse_list = []
loss_list = []

for train_index, test_index in kfold.split(x_train): 

    print(train_index.shape)
    print(test_index.shape)

    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]

    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=True ) 
    
    # 2. 모델구성

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(6,1))) 
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=52, kernel_size=2, padding='same', activation='relu')) 
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')) 
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=12, kernel_size=2, padding='same', activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(5))
    model.add(Dense(1))

    # 3. 컴파일 훈련

    modelpath = '../data/modelcheckpoint/team_LSTM2_'+str(num)+'.hdf5'
    es= EarlyStopping(monitor='val_loss', patience=10)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

    model.compile(loss='mse', optimizer='adam', metrics='mae')
    model.fit(x_train1, y_train1, epochs=100, batch_size=64, validation_data=(x_val,y_val), callbacks=[es,reduce_lr] )

    # 4. 평가, 예측

    loss, mae = model.evaluate(x_test1, y_test1, batch_size=64)
    y_predict = model.predict(x_pred)

    print(loss)

    # RMSE 
    print("RMSE : ", RMSE(y_pred, y_predict))

    # R2 만드는 법
    r2 = r2_score(y_pred, y_predict)
    print("R2 : ", r2)

    num += 1

    r2_list.append(r2_score(y_pred, y_predict))
    rmse_list.append(RMSE(y_pred, y_predict))
    loss_list.append(loss)


print("LSTM 윈도우 없음")
print("r2 : ",r2_list)
print("RMSE : ",rmse_list)
print("loss : ",loss_list)


