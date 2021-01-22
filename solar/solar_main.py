
# 해볼 것 

# 1. 댄스모델로 해볼 것 
# 유튜브에 잘 나온다던 모델로 변경해볼것 

# 2. 데이터 배열을 바꿔서 해볼 것



import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

#train = pd.read_csv('/content/drive/My Drive/solar/train/train.csv')
#submission = pd.read_csv('/content/drive/My Drive/solar/sample_submission.csv')


train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')


def make_cos(dataframe): # 특정 열이 해가 뜨고 해가지는 시간을 가지고 각 시간의 cos를 계산해주는 함수
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

def preprocess_data(data, is_train = True):
    a = pd.DataFrame()
    for i in range(int(len(data)/48)):
        tmp = pd.DataFrame()
        tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
        tmp['cos'] = make_cos(tmp)
        a = pd.concat([a,tmp])
    data['cos'] = a
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    
    # 새로운 TD, t-td 컬럼 추가 
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])

    temp = data.copy()
    temp = temp[['TARGET','GHI','DHI','DNI','T-Td','RH','T']]

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()

        return temp.iloc[:-96]

    elif is_train == False:
        temp = temp[['TARGET','GHI','DHI','DNI','T-Td','RH','T']]

        return temp.iloc[-48:, :]

df_train = preprocess_data(train)

# 상관계수
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark', font_scale=1.2, font='Malgun Gothic') # , palette='pastel'
sns.color_palette('Paired',6)
sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True)
plt.show()

x_train = df_train.to_numpy()

df_test = []
for i in range(81):
    file_path = '../data/solar/test/%d.csv'%i
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
x_test = x_test.to_numpy()


from sklearn.preprocessing import StandardScaler
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


print(x.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split as tts
x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x,y1,y2, train_size = 0.7,shuffle = True, random_state = 0)

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, Input,BatchNormalization,Activation

from tensorflow.keras.models import Sequential,  Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, Input, Reshape


def mymodel():

    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'swish',input_shape = (1,7)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'swish'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'swish'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'swish'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'swish'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation = 'swish'))
    model.add(Dense(32, activation = 'swish'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

    return model
    

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.3, verbose = 1)
epochs = 300
bs = 16


# 내일!!
x = []
for i in quantiles:
    model = mymodel()
    #filepath_cp = f'../dacon/data/modelcheckpoint/dacon_02_y1_quantile_{i:.1f}.hdf5'
    #cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y1_val),callbacks = [es,lr])

    print(model.predict(x_test))

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
        
#submission.to_csv('/content/drive/My Drive/solar/value/test_value_ver1.csv', index = False)
submission.to_csv('../data/solar/value/test_ver1.csv', index = False)
