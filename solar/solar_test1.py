
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')


def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

def preprocess_data(data, is_train = True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
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




def split_xy(dataset, time_steps, y_column, x_steps):
    x,y = list(), list()
    for i in range(len(dataset)):

        i = i*time_steps

        x_end_number = i + time_steps*x_steps
        y_end_number = x_end_number + time_steps*y_column

        if y_end_number > len(dataset):
            break

        temp_x = dataset[i:x_end_number,:]
        temp_y = dataset[x_end_number:y_end_number]

        x.append(temp_x)
        y.append(temp_y)

    return np.array(x), np.array(y)

x, y = split_xy(x_train,336,2,7)


print(x[0])
print(x[1])

'''
[[  0.   0.   0. ... -12.   0.   0.]
 [  0.   0.   0. ... -12.   0.   0.]
 [  1.   0.   0. ... -12.   0.   0.]
 ...
 [ 22.   0.   0. ...  -1.   0.   0.]
 [ 23.   0.   0. ...  -1.   0.   0.]
 [ 23.   0.   0. ...  -1.   0.   0.]]
'''


# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scale = StandardScaler()
# scale.fit(x_train[:,:-2])
# x_train[:,:-2] = scale.transform(x_train[:,:-2])
# x_test = scale.transform(x_test)


'''
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

def split_x(data,timestep):
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))

x,y1,y2 = split_xy(x_train,1)
x_test = split_x(x_test,1)


print(x[0])
'''

# [[  0.     0.     0.     0.     0.     1.5   69.08 -12.  ]]