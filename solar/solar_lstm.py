import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')

day = 7 

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

def split_to_seq(data): 
    tmp = []
    for i in range(48):
        tmp1 = pd.DataFrame()
        for j in range(int(len(data)/48)):
            tmp2 = data.iloc[j*48+i,:]
            tmp2 = tmp2.to_numpy()
            tmp2 = tmp2.reshape(1,tmp2.shape[0])
            tmp2 = pd.DataFrame(tmp2)
            tmp1 = pd.concat([tmp1,tmp2])
        x = tmp1.to_numpy()
        tmp.append(x)
    return np.array(tmp)

def make_cos(dataframe): 
    dataframe /=dataframe
    c = dataframe.dropna()
    d = c.to_numpy()

    def into_cosine(seq):
        for i in range(len(seq)):
            if i < len(seq)/2:
                seq[i] = float((len(seq)-1)/2) - (i)
            if i >= len(seq)/2:
                seq[i] = seq[len(seq) - i - 1]
        seq = seq/ np.max(seq) * np.pi/2
        seq = np.cos(seq)
        return seq

    d = into_cosine(d)
    dataframe = dataframe.replace(to_replace = np.NaN, value = 0)
    dataframe.loc[dataframe['cos'] == 1] = d
    return dataframe


# 상관 관계 
# 확인 해보니 'TARGET','GHI','DHI','DNI','T-Td' 타겟과 관련 높음 
# 'RH' 역관계 역관계도 크다면 데이터 구하는 것에 이득이 있을까? 의문 테스트 해볼 것


def preprocess_data(data, is_train = True):
    a = pd.DataFrame()
    for i in range(int(len(data)/48)):
        tmp = pd.DataFrame()
        tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
        tmp['cos'] = make_cos(tmp)
        a = pd.concat([a,tmp])
    data['cos'] = a
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['TARGET','GHI','DHI','DNI','RH','T-Td']]

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        return temp.iloc[-48*day:, :]

df_train = preprocess_data(train)
scale.fit(df_train.iloc[:,:-2])
df_train.iloc[:,:-2] = scale.transform(df_train.iloc[:,:-2])

df_test = []
for i in range(81):
    file_path = '../data/solar/test/%d.csv'%i
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp,is_train=False)
    temp = scale.transform(temp)
    temp = pd.DataFrame(temp)
    temp = split_to_seq(temp)
    df_test.append(temp)

test = np.array(df_test)
train = split_to_seq(df_train)

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

x,y1,y2 = [],[],[]
for i in range(48):
    tmp1,tmp2,tmp3 = split_xy(train[i],day)
    x.append(tmp1)
    y1.append(tmp2)
    y2.append(tmp3)

x = np.array(x) # (48, 훈련수, 일수, 6)
y1 = np.array(y1) # (48, 훈련수, 1)
y2 = np.array(y2) # (48, 훈련수, 1)

from sklearn.model_selection import train_test_split as tts

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D
def mymodel():
    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (day,6)))
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

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25, verbose = 1)
epochs = 10000
bs = 64

for i in range(48):
    # 시계열이라면 셔플을 안해야 시계열이지 않을까?
    x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x[i],y1[i],y2[i], train_size = 0.7,shuffle = False, random_state = 0)
    # 타겟 1
    for j in quantiles:
        model = mymodel()
        filepath_cp = f'../data/solar/modelcheckpoint/dacon_07_{i:2d}_y1seq_{j:.1f}.hdf5'
        cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        model.fit(x_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y1_val),callbacks = [es,cp,lr])

    # 타겟2
    for j in quantiles:
        model = mymodel()
        filepath_cp = f'../data/solar/modelcheckpoint/dacon_07_{i:2d}_y2seq_{j:.1f}.hdf5'
        cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        model.fit(x_train,y2_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y2_val),callbacks = [es,cp,lr]) 


# 셔플 ture 버전 끝나고 false 버전 돌릴 것