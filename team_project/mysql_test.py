
import db_connect as db
import numpy as np



# db 직접 불러오기 
'''
query = "SELECT * FROM time_location_data"
db.cur.execute(query)
select = np.array(db.cur.fetchall())

db.connect.commit()

print(select.shape)
'''

# 함수로 불러오기
print(db.load_table('time_location_data').shape)


