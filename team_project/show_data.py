import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pymysql
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

# 날짜별================================================================================================================================
connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "SELECT DATE,SUM(VALUE) FROM `time_location_data` WHERE si = '서울특별시' GROUP BY DATE" #DATE >= '2019-08-01' AND 
cur.execute(query)
select = np.array(cur.fetchall())
connect.commit()

x = select[65:,0]
y = select[65:,1]
'''
y = list(map(int, y))
plt.plot(x, y)
plt.title('Date')
plt.show()
'''
# 정규분포
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic" # 한글 변환
plt.figure(figsize=(10,10))
sns.distplot(y, rug=True,fit=norm) #distplot 히스토그램 정규분포
plt.title("주문량 분포도",size=15, weight='bold')
plt.show()


# 표준 정규분포
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic" # 한글 변환
plt.figure(figsize=(10,10))

print(y.shape)

y = y.reshape(-1,1)

print(y)

data_standadized_skl = StandardScaler().fit_transform(y)
print(data_standadized_skl)

sns.distplot(data_standadized_skl, rug=True,fit=norm) #distplot 히스토그램 정규분포
plt.title("주문량 분포도",size=15, weight='bold')
plt.show()


# 이상치 확인
import numpy as np
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))
outlier_loc = outliers(y)
print("이상치의 위치 : ",outlier_loc)

# Q-Q plot & boxplot
from scipy.stats import norm
from scipy import stats
fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
stats.probplot(y, plot=plt) 
green_diamond = dict(markerfacecolor='g', marker='D')
ax1.boxplot(y, flierprops=green_diamond)
plt.show()
