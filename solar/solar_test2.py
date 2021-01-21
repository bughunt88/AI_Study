import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('./practice/dacon/data/train/train.csv')
submission = pd.read_csv('./practice/dacon/data/sample_submission.csv')


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


a = pd.DataFrame()
for i in range(int(len(data)/48)):
    tmp = pd.DataFrame()
    tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
    tmp['cos'] = make_cos(tmp)
    a = pd.concat([a,tmp])
data['cos'] = a

print(train.info())