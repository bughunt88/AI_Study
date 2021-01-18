import io
import pandas as pd
import numpy as np

size = 4 #30

filename = '../data/삼성전자.csv'
filename2 = '../data/삼성전자2.csv'
filename3 = '../data/삼성전자3.csv'



df = pd.read_csv(filename, index_col=0,header=0,encoding='CP949')
df2 = pd.read_csv(filename2, index_col=0,header=0,encoding='CP949')
df3 = pd.read_csv(filename3, index_col=0,header=0,encoding='CP949')

df.replace(',','',inplace=True, regex=True)
df2.replace(',','',inplace=True, regex=True)
df3.replace(',','',inplace=True, regex=True)


'''
df = df.iloc[:662,:]
df = df.astype('float32')
# # 상관 계수 시각화!
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.5)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()
'''


# 액분 후 데이터
df = df.iloc[:662,[0,1,2,6,3]]
df2 = df2.iloc[0,[0,1,2,8,3]]
df3 = df3.iloc[0,[0,1,2,8,3]]



df = df.astype('float32')
df2 = df2.astype('float32')
df3 = df3.astype('float32')



df = df.sort_values(by=['일자'], axis=0)

df = df.append(df2)
df = df.append(df3)


total_data = df.to_numpy()


y_data = total_data




print(y_data[-2][0])


def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

total_data = split_x(total_data,size)


x = total_data[:-2,:size, :]

# = total_data[2:,size-1,-1:]

y_list = []

print(total_data.shape)

print(range(len(y_data) - size + 1))

for n in range(len(y_data) - size + 1):
  
  y_list.append([y_data[n+size-2][0],y_data[n+size-1][0]])

x_pred = total_data[-2,:size,:]

print(x[-4])
print(y_list[-4])