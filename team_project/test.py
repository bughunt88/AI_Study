# 시간2-8삭제 데이터로 conv1d 모델 결과 확인

import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM, Conv2D,Input,Activation, LSTM, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
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
    inputs = Input(shape = (x_train.shape[1],1),name = 'input')
    x = Conv1D(filters=1024,kernel_size=2,padding='same',strides = 2, activation=acti)(inputs)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(256, activation=acti)(x)
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

    return  score_list, y_predict

start_time = timeit.default_timer()


def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # pred = df.iloc[:,1:-1]

    if is_train == True:
        # train, test 나누기
        train_value = df[ '2020-09-01' > df['date'] ]
        x = train_value.iloc[:,1:-1].astype('int64')
        y = train_value.iloc[:,-1].astype('int64').to_numpy()
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1].astype('int64')
        y = test_value.iloc[:,-1].astype('int64').to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"]).to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y

x_train, y_train = load_data("SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC")
x_pred, y_pred = load_data("select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC",is_train=False)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1)

print(x_train.shape, y_train.shape) #(2459424, 42, 1) (2459424,)
print(x_pred.shape, y_pred.shape)   #(177408, 42, 1) (177408,)

leaky_relu = tf.nn.leaky_relu
acti = 'selu' #leaky_relu, mish, 'swish', 'elu', 'relu','tanh'
opti = RMSprop#, Nadam, Adam, Adadelta, Adamax, Adagrad, SGD]
batch = 200
lrr = 0.001
epo = 50

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 

modelpath = './mitzy/data/modelcheckpoint/19_deltime_data_Conv1D.hdf5'
model = build_model(acti, opti, lrr)
# model = load_model(modelpath) #, custom_objects={'leaky_relu':tf.nn.leaky_relu, 'mish':mish})

# 훈련
er,mo,lr = callbacks(modelpath) 
history = model.fit(x_train, y_train, verbose=1, batch_size=batch, epochs = epo, validation_data=(x_val,y_val), callbacks = [er, lr, mo])

score, y_predict = evaluate_list(model)

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))


# 엑셀 추가 코드 
# 경로 변경 필요!!!!

df = pd.DataFrame(y_predict)
df['test'] = y_pred
df.to_csv('./mitzy/data/csv/19_deltime_data_Conv1D.csv',index=False)




import matplotlib.pyplot as plt
 
fig = plt.figure( figsize = (12, 4))
chart = fig.add_subplot(1,1,1)
chart.plot(y_pred, marker='o', color='blue', label='Actual')
chart.plot(y_predict, marker='^', color='red', label='Predict')
plt.legend(loc = 'best') 
plt.show()

# r2 :  -0.0020812435389312434
# rmse :  3.8854697119962385
# mae :  1.3648083262994863
# mse :  15.096874882840133
# 5542.031775초 걸렸습니다.