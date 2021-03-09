
# 데이터 읽어서 분석해 주는 라이브러리

import pandas as pd
import pandas_profiling
df = pd.read_csv('titanic/train.csv')
profile = df.profile_report()
profile