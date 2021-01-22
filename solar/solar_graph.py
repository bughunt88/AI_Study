# 예측값으로 그래프 그리기      > 무진님꺼 따라하기 

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pandas import DataFrame

pred = pd.read_csv('../data/csv/dacon1/sub_1021_4.csv')

ranges = 336        # 7일치로 보겠음
hours = range(ranges)
pred = pred[ranges:ranges+ranges]

q_01 = pred['q_0.1'].values
q_02 = pred['q_0.2'].values
q_03 = pred['q_0.3'].values
q_04 = pred['q_0.4'].values
q_05 = pred['q_0.5'].values
q_06 = pred['q_0.6'].values
q_07 = pred['q_0.7'].values
q_08 = pred['q_0.8'].values
q_09 = pred['q_0.9'].values

import matplotlib.pyplot as plt

plt.figure(figsize=(18,2.5))
plt.subplot(1,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='#aa00cc')
plt.plot(hours, q_03, color='#00ccaa')
plt.plot(hours, q_04, color='#ccaa00')
plt.plot(hours, q_05, color='#00aacc')
plt.plot(hours, q_06, color='#aacc00')
plt.plot(hours, q_07, color='#cc00aa')
plt.plot(hours, q_08, color='#000000')
plt.plot(hours, q_09, color='blue')
plt.legend()
plt.show()