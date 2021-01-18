import numpy as np
import pandas as pd


# 데이터 변수
size = 30 #30


kodex_data = pd.read_csv('../data/kodex.csv', index_col=0,header=0,encoding='CP949')
kodex_data.replace(',','',inplace=True, regex=True)


kodex_data.drop(['전일비'], axis='columns', inplace=True)
kodex_data = kodex_data.astype('float32')
kodex_data = kodex_data.iloc[:662,:]
kodex_data = kodex_data.iloc[:,[6,7,8,10,11,12]]

kodex_data = kodex_data.sort_values(by=['일자'], axis=0)


total_kodex_data = kodex_data.to_numpy()


def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

total_kodex_data  = split_x(total_kodex_data,size)


'''
# 상관 계수 시각화!
import matplotlib.pyplot as plt
import seaborn as sns

kodex_data = pd.read_csv('../data/test.csv', index_col=0,header=0,encoding='CP949')
kodex_data.replace(',','',inplace=True, regex=True)
kodex_data = kodex_data.astype('float32')

sns.set(font_scale=1)
sns.heatmap(data=kodex_data.corr(), square=True, annot=True, cbar=True)
plt.show()
'''

