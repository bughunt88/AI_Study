
# 데이터 일단위로 잘라서 시계열 만들기 



import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')

# def Add_features(data):
#     data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
#     data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
#     data.drop(['cos'], axis= 1, inplace = True)
#     return data

def preprocess_data(data, is_train = True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train == True:
        #temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        #temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()

        return temp.iloc[:-96]

    elif is_train == False:
        temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

        return temp.iloc[-48:, :]

df_train = preprocess_data(train)
x_train = df_train.to_numpy()

df_test = []
for i in range(81):
    file_path = '../data/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
x_test = x_test.to_numpy()
# x_test.shape = (3888, 8) ## 81일간 하루에 48시간씩 총 8 개의 컬럼 << 이걸 프레딕트 하면 81일간 48시간마다 2개의 컬럼(내일,모레)



from sklearn.preprocessing import MinMaxScaler, StandardScaler
scale = StandardScaler()
scale.fit(x_train[:,:-2])
x_train[:,:-2] = scale.transform(x_train[:,:-2])
x_test = scale.transform(x_test)

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

x,y1,y2 = split_xy(x_train,1)

def split_x(data,timestep):
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))

x_test = split_x(x_test,1)
# print(x.shape,y1.shape,y2.shape) # (52464, 1, 8) (52464, 1) (52464, 1) >> 한 시간대에 x행으로 다음날, 모레 같은 시간대의 타겟
# y1 을 내일의 타겟, y2 를 모레의 타겟!!

from sklearn.model_selection import train_test_split as tts
x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x,y1,y2, train_size = 0.7,shuffle = False, random_state = 0)

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D

def mymodel():
    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (1,8)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))
    return model

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.3, verbose = 1)
epochs = 1000
bs = 32


# 내일!!
x = []
for i in quantiles:
    model = mymodel()
    #filepath_cp = f'../dacon/data/modelcheckpoint/dacon_02_y1_quantile_{i:.1f}.hdf5'
    #cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y1_val),callbacks = [es,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

# 모레!!
x = []
for i in quantiles:
    model = mymodel()
    #filepath_cp = f'../dacon/data/modelcheckpoint/dacon_02_y2_quantile_{i:.1f}.hdf5'
    #cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y2_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y2_val),callbacks = [es,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp2 = pd.concat(x, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2
        
submission.to_csv('../data/solar/value/0120_trynew.csv', index = False)

