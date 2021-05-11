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
from tensorflow.keras.applications import VGG16, MobileNet, ResNet101, EfficientNetB5, EfficientNetB7

# db 직접 불러오기 

# 0 있다
query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC "

db.cur.execute(query)
dataset = np.array(db.cur.fetchall())

# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

print(df.shape)


# 원 핫으로 컬럼 추가해주는 코드!!!!!
df = pd.get_dummies(df, columns=["category", "dong"])
# 카테고리랑 동만 원핫으로 해준다 



# train, test 나누기

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1,1)
# y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1],1)




print(df.shape)





