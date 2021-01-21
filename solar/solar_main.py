import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()

        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)



df_test = []
for i in range(81):
    file_path = '../data/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)

    temp = preprocess_data(temp, is_train=False)

    #if i == 0:
        #print(i)
        #print(temp)

    
    df_test.append(temp)

X_test = pd.concat(df_test)


x_predict_test = []
file_path = '../data/solar/test/0.csv'
temp = pd.read_csv(file_path)
temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
temp = temp.iloc[:48, :]
x_predict_test.append(temp)


from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], shuffle=False, test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], shuffle=False, test_size=0.3, random_state=0)


print(X_train_1.shape)
print(X_train_1)

print("######")

print(Y_train_1.shape)



#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,Input,Activation, LSTM, Reshape, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum


early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

def quantile_loss(q, y, pred):
  err = (y-pred)
  
  return mean(maximum(q*err, (q-1)*err), axis=-1)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


predict_list = []



for q in q_lst:

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(7,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1)) 
  
    model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam')
    model.fit(X_train_1,Y_train_1, epochs=1, batch_size=32, validation_data=(X_valid_1,Y_valid_1),callbacks=[early_stopping])
    predict_list.append(model.predict(X_test))
    print(model.predict(X_test).round(2))

predict_list = np.array(predict_list)

#print(predict_list.shape)

'''
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train_1,Y_train_1, epochs=100, batch_size=16, validation_data=(X_valid_1,Y_valid_1),callbacks=[early_stopping,reduce_lr])

'''


# 4. 평가, 예측

loss = model.evaluate(X_valid_1, Y_valid_1, batch_size=32)
y_predict = model.predict(X_valid_1)






from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

print('loss : ', loss)
print("RMSE : ", RMSE(Y_valid_1, y_predict))

# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(Y_valid_1, y_predict)
print("R2 : ", r2)



y_predict = model.predict(x_predict_test)
print(y_predict)

