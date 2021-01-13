import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)

# index_col, header 
# 인덱스와 헤더를 명시해준다 
# 인덱스 디폴트는 none, 헤더는 0 이다
# 헤더가 없으면 None를 넣어줘야 한다 

print(df)

