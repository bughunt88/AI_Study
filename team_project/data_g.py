import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

import pymysql
'''
# # 날짜별
connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "SELECT DATE,SUM(VALUE) FROM `time_location_data` WHERE DATE >= '2019-08-01' AND si = '서울특별시' GROUP BY DATE"
cur.execute(query)
select = np.array(cur.fetchall())
connect.commit()

x = select[:,0]
y = select[:,1]

plt.plot(x, y)
plt.show()

# # 시간별
connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "SELECT TIME,SUM(VALUE) FROM `time_location_data` WHERE DATE >= '2019-08-01' AND si = '서울특별시' GROUP BY TIME ORDER BY TIME ASC"
cur.execute(query)
select = np.array(cur.fetchall())
connect.commit()

x = select[:,0]
y = select[:,1]

plt.plot(x, y)
plt.show()

# # 월별
connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "SELECT DATE_FORMAT(DATE,'%Y-%m'), SUM(VALUE) FROM `time_location_data` WHERE DATE >= '2019-08-01' AND si='서울특별시' GROUP BY DATE_FORMAT(DATE,'%Y-%m');"
cur.execute(query)
select = np.array(cur.fetchall())
connect.commit()
print(select)

x = select[:,0]
y = select[:,1]

plt.plot(x, y)
plt.show()
'''

# # 업종별
connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "SELECT DATE_FORMAT(DATE,'%Y-%m') AS date_v, category, SUM(VALUE) FROM `business_location_data` WHERE si = '서울특별시' AND (category = '치킨' OR  category = '한식'  OR  category = '분식' OR  category = '카페/디저트' OR  category = '족발/보쌈') GROUP BY DATE_FORMAT(DATE,'%Y-%m'), category ORDER BY date_v, category DESC"
cur.execute(query)
select = np.array(cur.fetchall())
connect.commit()
print(select)

x = select[:,0]
y = select[:,1]

plt.plot(x, y)
plt.show()


