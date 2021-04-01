import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

# 자료불러오기
df = pd.read_csv('C:\order_project\시간-지역별 배달 주문건수_20201031000000.csv',thousands = ',', index_col=0, header=None)
print(df.head())

is_seoul = df[2] == '서울특별시'
Seoul = df[is_seoul]
print(Seoul.head())
# Seoul.to_csv('C:/order_project/seoul_data.csv',index=True)

# # 날짜별
# Seoul = Seoul.drop(Seoul.columns[[0,1,2,3]], axis='columns')
# print(Seoul.head())
# date_sum = Seoul.groupby(0).sum()
# print(date_sum.head())

# # 시간별
# date_sum = Seoul.groupby(1).sum()
# print(date_sum.head())

# 월별
Seoul = Seoul.drop(Seoul.columns[[0,1,2,3]], axis='columns')
print(Seoul.head())
# Seoul['2019-08-01':'2019-08-31']
date_sum = []
for m in range(1,13):
    a = Seoul['2019-'+"%02d"%m+'-01':'2019-'+"%02d"%m+'-30'].sum()
    a.index=[m]
    date_sum.append(a[:])

col_name = ['colname1', 'colname2']
list1 = date_sum
list_df = pd.DataFrame(list1, columns=col_name)
print(list1)
print(list_df)


import matplotlib.pyplot as plt
import matplotlib

plt.figure()
ax = date_sum.plot.line()
plt.show()