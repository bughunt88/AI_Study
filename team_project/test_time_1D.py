import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation, LSTM, Reshape
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
import tensorflow.keras.backend as K

# db 직접 불러오기 
total_score_list = []


def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def mish(x):
    return x * K.tanh(K.softplus(x))

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

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr

def build_model(acti, opti, lr):   

    # 2. 모델구성
    inputs = Input(shape = (x_train.shape[1],x_train.shape[2]),name = 'input')
    x = Conv1D(filters=1024, kernel_size=2, padding='same', strides=2, activation=acti)(inputs)
    x = Conv1D(filters=1024, kernel_size=2, padding='same', strides=2, activation=acti)(x)
    x = Flatten()(x)
    #x = Dense(1024, activation=acti)(x)
    x = Dense(512, activation=acti)(x)
    # x = Reshape((x_train1.shape[1],x_train1.shape[2]))(x)
    outputs = Dense(2688)(x)
    model = Model(inputs=inputs,outputs=outputs)

    # 3. 컴파일 훈련        
    model.compile(loss='mse', optimizer = opti(learning_rate=lr), metrics='mae')

    return model

def evaluate_list(model):
    score_list = []
    #4. 평가, 예측
    y_predict = model.predict(x_pred)

    # r2_list
    r2 = r2_score(y_pred, y_predict)
    score_list.append(r2)
    print('r2 : ', r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    score_list.append(rmse)
    print('rmse : ', rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    score_list.append(mae)
    print('mae : ', mae)
    # mse_list
    mse = mse_(y_pred, y_predict, squared=True)
    score_list.append(mse)
    print('mse : ', mse)

    return  score_list

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

    kfold = KFold(n_splits=3, shuffle=False)

    x_train, y_train = split_xy(train_value, 2688, 2688)
    x_pred, y_pred = split_xy(test_value, 2688, 2688)

    num = 0 

    # print(x_train.shape, y_train.shape)    #(398, 2688, 6) (398, 2688)
    # print(x_pred.shape, y_pred.shape)     #(8, 2688, 6) (8, 2688)

    print("###########")
    print(x_train.shape[1],x_train.shape[2])
    print("###########")

    for train_index, test_index in kfold.split(x_train): 

        x_train1, x_test1 = x_train[train_index], x_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]
        
        x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=True ) 

        
        # 모델
        acti_list = ['swish', 'elu', 'relu', 'selu','tanh']
        opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad, SGD]


        acti = acti_list[3]
        opti = opti_list[2]


        batch = 64
        lrr = 0.01
        epo = 1000

        model = build_model(acti, opti, lrr)

        # 훈련
        modelpath = '../data/modelcheckpoint/team_1D_'+str(num)+'.hdf5'
        er,mo,lr = callbacks(modelpath) 
        history = model.fit(x_train1, y_train1, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])

        # 4. 평가, 예측
        # loss, mae = model.evaluate(x_test1, y_test1, batch_size=batch)
        # y_predict = model.predict(x_pred)

        score = evaluate_list(model)
        total_score_list.append(score)
        print(f'============{num}fold=================')
        print('r2   : ', score[0])
        print('rmse : ', score[1])
        print('mae : ', score[2])
        print('mse : ', score[3])
        print(f'======================================')

        num += 1


    # print("r2 : ",r2_list)
    # print("RMSE : ",rmse_list)
    # print("loss : ",loss_list)

    # main_r2_list.append(r2_list)
    # main_rmse_list.append(rmse_list)
    # main_loss_list.append(loss_list)

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

print('=========================final score========================')
print("r2    rmse   mae     mse: ")
print(total_score_list)