import numpy as np
import pandas as pd


df = pd.read_csv("../data/삼성전자.csv",encoding='cp949',index_col=0)
df_1 = df.iloc[:662,:]
df_2 = df.iloc[665:,:]

df = pd.concat([df_1,df_2])
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')

df.iloc[662:,0:4] = df.iloc[662:,0:4]/50.0
df.iloc[662:,5:] = df.iloc[662:,5:]*50


# df.drop(['종가', '등락률', '거래량' ,'금액(백만)','신용비','개인','외인(수량)','외국계','외인비'], axis='columns', inplace=True)
# 상관 관계 50 먹이기 전 (기관, 프로그램)

df.drop(['등락률', '기관' ,'금액(백만)','신용비','프로그램','외인(수량)','외국계','외인비'], axis='columns', inplace=True)
# 상관 관계 50 먹이기 후 (거래량, 개인)

df = df.sort_values(by=['일자'], axis=0)

print(df)


