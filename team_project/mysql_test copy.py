
import db_connect as db
import numpy as np

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

query = "SELECT DATE, SUM(VALUE) FROM `business_location_data` WHERE si='서울특별시' AND (DATE >= '2019-07-17' AND '2019-09-30' >= DATE ) GROUP BY DATE ORDER BY DATE ASC"
db.cur.execute(query)
select1 = np.array(db.cur.fetchall())
#select = db.cur.fetchall()

query = "SELECT DATE, SUM(VALUE) FROM `business_location_data` WHERE si='서울특별시' AND (DATE >= '2020-07-17' AND '2020-09-30' >= DATE ) GROUP BY DATE ORDER BY DATE ASC"
db.cur.execute(query)
select2 = np.array(db.cur.fetchall())

db.connect.commit()

x1 = select1[:-4,0]
y1 = select1[:-4,1]

x2 = select2[:,0]
y2 = select2[:,1]

print(x1.shape)
print(y1.shape)

print(x2.shape)
print(y2.shape)

import matplotlib.pyplot as plt
import matplotlib

plt.figure(figsize=(10,6))
plt.plot(x1, y1)
plt.plot(x1, y2)
plt.show()