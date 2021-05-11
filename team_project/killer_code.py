  
import numpy as np
import db_connect as db
import pandas as pd
import warnings
import joblib
import timeit
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.activations import tanh, relu, elu, selu, swish
from tensorflow.keras.layers import LeakyReLU, PReLU
from time import time


# db 직접 불러오기 =================================================

# 0 없다
# query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
# WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
# DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"

# 0 있다
query = "SELECT * FROM `main_data_table`"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기
column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']
df = pd.DataFrame(dataset, columns=column_name)
db.connect.commit()


# train, test 나누기
pred = df.iloc[:,1:-1]
train_value = df[ '2020-09-01' > df['date'] ]
x_train1 = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train1 = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']
x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

# print(x_train1.shape, y_train1.shape) #(3472128, 6) (3472128,)
# print(x_pred.shape, y_pred.shape)   #(177408, 6) (177408,)
#=======================================================================

def build_model(drop=0.5, optimizer=RMSprop, filters=100, kernel_size=2, learning_rate=0.1, activation = PReLU):
    inputs = Input(shape = (6,1),name = 'input')
    x = Conv1D(filters=filters,kernel_size=kernel_size,padding='same',activation=activation(),name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv1D(filters=filters,kernel_size=kernel_size,padding='same',activation=activation(),name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv1D(filters=filters,kernel_size=kernel_size,padding='same',activation=activation(),name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1,name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'mse',optimizer = optimizer(learning_rate=learning_rate), metrics = ['mae'])
    return model

def create_hyperparameter() : 
    batchs = [50, 60, 70, 80]
    optimizers = [RMSprop, Adam, Adadelta, Adamax, Adagrad, SGD, Nadam]
    dropout = [0.1, 0.2, 0.3]
    filters = [10,50,100,200,300]
    kernel_size = [2, 3]
    activations = [tanh, relu, elu, selu, swish,LeakyReLU, PReLU]
    learning_rate = [0.1, 0.005, 0.001]
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop': dropout, 'filters':filters,
    'kernel_size' : kernel_size, 'learning_rate' : learning_rate, 'activation':activations}  

hyperparameters = create_hyperparameter()
model2 = KerasRegressor(build_fn=build_model, verbose = 1)   #, epochs = 2)
search = GridSearchCV(model2, hyperparameters, cv=kfold)


def callbacks(modelpath):
    er = EarlyStopping(monitor = 'val_loss',patience=30)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=15, factor=0.5)
    return er,mo,lr

kfold = KFold(n_splits=5, shuffle=True)
hyperparameters = create_hyperparameter()
# model = build_model()

num = 0 

r2_list = []
rmse_list = []
mae_list = []
time_list = []
best_estimator_list = []

# 훈련 loop
for train_index, valid_index in kfold.split(x_train1):       

    # print(train_index, len(train_index))    #2777702
    # print(valid_index, len(valid_index))    #694426

    x_train = x_train1[train_index]
    x_valid = x_train1[valid_index]
    y_train = y_train1[train_index]
    y_valid = y_train1[valid_index]

    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_valid=x_valid.reshape(x_valid.shape[0], x_valid.shape[1],1)
    
    #2. 모델구성
    model2 = KerasRegressor(build_fn=build_model, verbose = 1)   #, epochs = 2)
    search = GridSearchCV(model2, hyperparameters, cv=kfold)

    start_time = timeit.default_timer()
    #3. 훈련
    modelpath =f'./data/hdf5/1_conv1d_kfold_{num}.hdf5'
    er,mo,lr = callbacks(modelpath) 
    search.fit(x_train, y_train, verbose=1, epochs = 200, validation_data=(x_valid, y_valid), callbacks = [er, lr, mo])

    finish_time = timeit.default_timer()
    time = round(finish_time - start_time, 2)
    time_list.append(time)
    print(f'{num}fold time : ', time)

    # best_estimator_
    print(f'{num}fold 최적의 매개변수 : ', search.best_estimator_) 
    best_estimator_list.append(search.best_estimator_) 

    # 모델저장
    joblib.dump(search.best_estimator_, f'./data/h5/1_conv1d_kfold_{num}.pkl')

    # 모델로드
    model = joblib.load(f'./data/h5/1_conv1d_kfold_{num}.pkl')

    #4. 평가, 예측
    y_predict = model.predict(x_pred)
    print('예측값 : ', y_predict[:5])
    print('실제값 : ', y_pred[:5])


    # r2_list
    r2 = r2_score(y_pred, y_predict)
    print(f'{num}fold r2 score     :', r2)
    r2_list.append(r2)
    # rmse_list
    rmse = mse_(y_pred, y_predict, squared=False)
    print(f'{num}fold rmse score     :', rmse)
    rmse_list.append(rmse)
    # mae_list
    mae = mae_(y_pred, y_predict)
    print(f'{num}fold mae score     :', mae)
    mae_list.append(mae) 

    num += 1

r2_list = np.array(r2_list)
print('r2_list : ', r2_list)
rmse_list = np.array(rmse_list)
print('rmse_list : ',rmse_list)
mae_list = np.array(mae_list)
print('mae_list : ',mae_list)
time_list = np.array(time_list)
print('time_list : ', time_list)
best_estimator_list = np.array(best_estimator_list)
print('best_estimator_list : ',best_estimator_list)

# 0fold time :  3885.64
# 하루동안 1fold도 못끝냄
# 정지