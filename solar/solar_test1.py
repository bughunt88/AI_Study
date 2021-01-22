
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')

# 판다스 데이터 프레임에 값 넣기 
# df = pd.DataFrame(train)

#행으로 찾기
# print(df.loc[[0,2]])


# 판다스 필터 방법
# 컬럼의 값을 기준으로 필터 하는 법 
# print(df[(df.T>0) & (df.Hour == 0)])


# 인덱스로 필터하는 법
# df.iloc[:,0:2]
       #행, 열 

# print(df[['T','RH']])

# like로 컬럼의 이름 찾아서 할 수 있다
# axis=1은 컬럼을 의미 
# print(df.filter(like='T', axis=1))


for i in range(48):
    print(train.loc[i*48])