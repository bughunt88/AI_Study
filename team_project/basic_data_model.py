import numpy as np

import db_connect as db
import pandas as pd


# db 직접 불러오기 

query = "SELECT * FROM main_data_table"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()


# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

x_train = train_value.iloc[0:,:-1]
y_train = train_value.iloc[0:,-1]

test_value = df[df['date'] >=  '2020-09-01']

x_test = test_value.iloc[0:,:-1]
y_test = test_value.iloc[0:,-1]



print(x_test)
print(x_train)
