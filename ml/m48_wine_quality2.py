
import numpy as np
import pandas as pd

df = pd.read_csv('../data/naver/winequality-white.csv', sep=';')

count_data = df.groupby('quality')['quality'].count()

print(count_data)


import matplotlib.pyplot as plt
count_data.plot()
plt.show()

