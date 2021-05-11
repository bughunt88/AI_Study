import numpy as np
import db_connect as db
import pandas as pd
import timeit
import tensorflow as tf

# 0 있다
query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC "


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

print(df)

df.iloc[:,-1] = df.iloc[:,-1].astype('int32')
df['category'] = df['category'].astype('int32')
df['dong'] = df['dong'].astype('int32')

print(df)


# 원 핫으로 컬럼 추가해주는 코드!!!!!

# df1 = pd.get_dummies(df, columns=["category", "dong"])
# print(df1)

from category_encoders import HashingEncoder
has = HashingEncoder(n_components=5)
category = has.fit_transform(df['category'])
dong = has.fit_transform(df['dong'])

print(df)

print(category)
print(dong)