# 시계열 데이터, 회기형 데이터에 결측치 처리 방법!!! (보간법)


from pandas import DataFrame, Series
from datetime import datetime
import numpy as np 
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021','3/3/2021','3/4/2021','3/5/2021']
dates = pd.to_datetime(datestrs)

print(dates)
print("====================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
