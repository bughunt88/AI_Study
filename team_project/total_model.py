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
fold_score_list = []
history_list=[]


def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def mish(x):
    return x * K.tanh(K.softplus(x))

def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=20)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=10, factor=0.3, verbose=1)
    return er,mo,lr

def build_model(acti, opti, lr):   

    # 2. 모델구성
    inputs = Input(shape = (x_train.shape[1]),name = 'input')
    x = Dense(1024,activation=acti)(inputs)
    x = Dropout(0.2)(x)
    x = Dense(256,activation=acti)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation=acti)(x)
    x = Dense(16, activation=acti)(x)
    outputs = Dense(1)(x)
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



'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

# 0 있다
query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC "


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# 원 핫으로 컬럼 추가해주는 코드!!!!!
df = pd.get_dummies(df, columns=["category", "dong"])
# 카테고리랑 동만 원핫으로 해준다 


# train, test 나누기
pred = df.iloc[:,1:-1]
train_value = df[ '2020-09-01' > df['date'] ]
x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']
x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()


fold = 3
kfold = KFold(n_splits=fold, shuffle=True)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)

print(x_train.shape, y_train.shape) 
print(x_pred.shape, y_pred.shape)   

leaky_relu = tf.nn.leaky_relu
acti_list = [leaky_relu, mish, 'swish', 'elu', 'relu', 'selu','tanh']
opti_list = [RMSprop, Nadam, Adam, Adadelta, Adamax, Adagrad, SGD]
batch = 200
lrr = 0.01
epo = 50
for op_idx,opti in enumerate(opti_list):
    for ac_idx,acti in enumerate(acti_list):
        num = 0 
        for train_index, test_index in kfold.split(x_train):             

            x_train1, x_test1 = x_train[train_index], x_train[test_index]
            y_train1, y_test1 = y_train[train_index], y_train[test_index]
            
            x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=True ) 

            model = build_model(acti, opti, lrr)

            # 훈련
            modelpath = f'../data/modelcheckpoint/15_models_compare_{op_idx}_{ac_idx}_fold' + str(num) + '.hdf5'
            er,mo,lr = callbacks(modelpath) 
            history = model.fit(x_train1, y_train1, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])
            # history_list.append(history)

            score = evaluate_list(model)
            fold_score_list.append(score)
            print('=============parameter=================')
            print('optimizer : ', opti, '\n activation : ', acti, '\n batch_size : ', batch, '\n lr : ', lrr, '\n epochs : ', epo)
            print(f'============{num}fold=================')
            print('r2   : ', score[0])
            print('rmse : ', score[1])
            print('mae : ', score[2])
            print('mse : ', score[3])
            print(f'======================================')

            num += 1

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

print('=========================final score========================')
print("r2           rmse          mae            mse: ")
fold_score_list = np.array(fold_score_list).reshape(len(opti_list),len(acti_list),fold,4)
print(fold_score_list)

""" 
# list all data in history==============================
import matplotlib.pyplot as plt
print(history.history.keys())

fig = plt.figure(figsize=(100,30))

for i in range(15):
    n = int(i/3)
    ax = fig.add_subplot(5,3,i+1)
    ax.plot(history_list[i].history["loss"])
    ax.plot(history_list[i].history["val_loss"])
    ax.plot(history_list[i].history["mae"])
    ax.plot(history_list[i].history["val_mae"])
    ax.set_title(f'{acti_list[n]}')
    ax.legend(['loss', 'val loss', 'mae', 'val mae'])
fig.tight_layout(h_pad = 8, w_pad = 8)
plt.show()
#===================================================== 
# """