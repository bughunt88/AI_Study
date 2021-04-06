import pandas as pd
import pymysql



connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "SELECT DATE,TIME,category, dong, VALUE FROM `business_location_data` WHERE DATE >= '2019-08-01' AND si = '서울특별시'"
cur.execute(query)
select = pd.DataFrame(cur.fetchall())
connect.commit()

select.columns = ['date','time','category','dong','value']
print(select)
# select = pd.DataFrame(select)

import missingno
missingno.matrix(select)
missingno.heatmap(select)
missingno.dendrogram(select)
missingno.bar(select)

plt.show()