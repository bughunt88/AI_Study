import numpy as np
import pandas as pd


t_data = pd.read_csv('../data/titanic/test.csv', index_col=0,header=0,encoding='CP949')
#t_data.replace(',','',inplace=True, regex=True)

t_main_data = t_data.iloc[:,[0,2,3,4,5,7]]
t_main_data['Sex'] = np.where(t_main_data['Sex'] != 'male', 0, 1)

t_main_data["Fare"].fillna(t_main_data.groupby(["Pclass"])["Fare"].transform("median"), inplace=True)
# 티켓 등급 기준으로 요금 평균

# 이 부분은 수정해야한다!!!!!
t_main_data["Age"].fillna(t_main_data.groupby(["Pclass"])["Age"].transform("median"), inplace=True)
# 요금 기준으로 나이 평균

t_main_data = t_main_data.astype({"Age": "int64"})

print(t_main_data)
print(t_main_data.info())





