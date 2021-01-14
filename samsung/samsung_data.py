import numpy as np
import pandas as pd


size = 2

df = pd.read_csv('../data/삼성전자.csv', index_col=0,header=0,encoding='CP949')
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')


# 액분 전 데이터

df = df.iloc[:662,:]
df.drop(['등락률', '기관' ,'프로그램','신용비','개인','외인(수량)','외국계','외인비'], axis='columns', inplace=True)

# 상관 관계 50 먹이기 전 (기관, 프로그램)



# 액분 후 데이터 
'''
df_1 = df.iloc[:662,:]
df_2 = df.iloc[665:,:]
df = pd.concat([df_1,df_2])
df.iloc[662:,0:4] = df.iloc[662:,0:4]/50.0
df.iloc[662:,5:] = df.iloc[662:,5:]*50
df.drop(['등락률', '기관' ,'금액(백만)','신용비','프로그램','외인(수량)','외국계','외인비'], axis='columns', inplace=True)
'''
# 상관 관계 50 먹이기 후 (거래량, 개인)

df = df.sort_values(by=['일자'], axis=0)

print(df)

df_x = df.iloc[:,[0,1,2,4,5,3]]

#df_y = df.iloc[size:,[3]]
#df_x_pred = df.iloc[-size:,[0,1,2,4,5]]

x_data = df_x.to_numpy()


def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

total_data = split_x(x_data,size)


# npy 저장
# np.save('../Study/samsung/samsung_data.npy', arr=total_data)



x = total_data[:-1,:size, :-1]
y = total_data[1:,size-1,-1:]
x_pred = total_data[-1,:size,:-1]
print(x_pred)
print(x_pred.shape)

# # 상관 계수 시각화!
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

