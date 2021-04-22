import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 

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

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']


x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1,1)
# y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1],1)

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

kfold = KFold(n_splits=3, shuffle=True)


start_time = timeit.default_timer()

num = 0 
r2_list = []
rmse_list = []
loss_list = []

leaky_relu = tf.nn.leaky_relu

for train_index, test_index in kfold.split(x_train): 

    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]

    x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1],1, 1)
    x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1],1, 1)

    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1,  train_size=0.9, random_state = 77, shuffle=False ) 

    # 2. 모델구성

    model=Sequential()
    model.add(Conv2D(filters=1024, kernel_size=(2,2), padding='same',
                    strides=(2,2) , input_shape=(6,1,1), activation=leaky_relu))
    model.add(Conv2D(filters=512, kernel_size=(2,2), padding='same',
                    strides=(2,2), activation=leaky_relu))
    model.add(Flatten())
    model.add(Dense(512, activation=leaky_relu))
    model.add(Dense(1)) 

    # 3. 컴파일 훈련

    modelpath = '../data/modelcheckpoint/team_2d_2_'+str(num)+'.hdf5'
    es= EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    cp =ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto')

    model.compile(loss='mse', optimizer='Nadam', metrics='mae')
    model.fit(x_train1, y_train1, epochs=1000, batch_size=64, validation_data=(x_val,y_val), callbacks=[es,reduce_lr,cp] )

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
LSTM 윈도우 없음 True
r2 :  [-0.0002796697158140571, -0.0004090043058784243, -0.000370540668791719]
RMSE :  [3.881975429677031, 3.8822263882278, 3.8821517557616914]
loss :  [12.300849914550781, 12.529807090759277, 12.243167877197266]

LSTM 윈도우 없음 0.9 false
r2 :  [-0.0005065010569298423, -0.0004432271097196683, -0.00040907636004416936]
RMSE :  [3.8824155584754445, 3.882292790836807, 3.882226528035907]
loss :  [12.459065437316895, 12.402216911315918, 12.2125883102417]
'''