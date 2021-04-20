import numpy as np
import db_connect as db
import pandas as pd
import timeit

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 

#size = 2688 #30

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def split_xy(dataset, time_steps, y_c):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end = i*384 + time_steps
        y_end = x_end + y_c

        if y_end > len(dataset):
            break
        tmp_x = dataset[i*384:x_end, :-1]
        tmp_y = dataset[x_end:y_end, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

start_time = timeit.default_timer()

for main_num in range(1):

    '''
    query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
    WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
    DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
    '''
    query = 'SELECT * FROM main_data_table WHERE dong = "' + str(main_num) + '" ORDER BY DATE, YEAR, MONTH ,TIME, category ASC'

    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기

    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

    df = pd.DataFrame(dataset, columns=column_name)

    db.connect.commit()

    # train, test 나누기

    train_value = df[ '2020-08-31' >= df['date'] ]
    train_value = train_value.iloc[:,1:].astype('int64').to_numpy()
    # x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
    # y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

    test_value = df[df['date'] >=  '2020-09-01']
    test_value = test_value.iloc[:,1:].astype('int64').to_numpy()

    # x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
    # y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()


    kfold = KFold(n_splits=3, shuffle=True)

    x_train, y_train = split_xy(train_value, 2688, 2688)
    x_pred, y_pred = split_xy(test_value, 2688, 2688)

    print(x_train.shape)
    print(y_train.shape)

    print(y_pred.shape)


    print("@@@@@")

    print(x_pred.shape)
    print(y_pred.shape)


    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    # x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

    num = 0 
    r2_list = []
    rmse_list = []
    loss_list = []

    for train_index, test_index in kfold.split(x_train): 

        x_train1, x_test1 = x_train[train_index], x_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]

        x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=True ) 
        
        # 2. 모델구성

        model = Sequential()
        model.add(LSTM(128, activation='relu' ,input_shape=(x_train1.shape[1],x_train1.shape[2]))) 
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(1)) 

        # 3. 컴파일 훈련

        modelpath = '../data/modelcheckpoint/team_LSTM1_'+str(num)+'.hdf5'
        es= EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
       #cp =ModelCheckpoint(filepath='modelpath', save_best_only=True)

        mc = ModelCheckpoint('../data/modelcheckpoint/lotte_0317_2.h5',save_best_only=True, verbose=1)

        model.compile(loss='mse', optimizer='adam', metrics='mae')
        model.fit(x_train1, y_train1, epochs=1000, batch_size=64, validation_data=(x_val,y_val), callbacks=[es,reduce_lr,mc] )

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


terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))


'''
LSTM 2개 윈도우 없음
r2 :  [-0.0006272980157535635, -0.0001606461551999505, 0.7029568019231744]
RMSE :  [3.8826499246874193, 3.881744464129382, 2.1154456272782514]
loss :  [12.228251457214355, 12.480032920837402, 2.2087907791137695]
'''

