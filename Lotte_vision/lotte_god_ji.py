import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from collections import Counter

num = 5
x = []
for i in range(num):           # 파일의 갯수
    df = pd.read_csv(f'C:/data/lotte/counter_csv/answer{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

# print(x.shape)
a1= []
a2= []
a3= []
df = pd.read_csv(f'C:/data/lotte/mode_csv/answer{i}.csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(num):         # 파일의 갯수
            b.append(x[k,i,j].astype('int'))
        # ======================================
        max1 = []
        max2 = []
        max3 = []

        count = Counter(b)

        # print(max(count,key=count.get))   #1개
        max_list = [k for k,v in count.items() if max(count.values()) == v] # 여러개

        max1 = max_list[0]
        a1.append(max1)

        # 세번째 어펜드
        if(len(max_list) > 2 ):
            max3 = max_list[2]
            a3.append(max3)
        else:
            max1 = max_list[0]
            a3.append(max1)

        # 두번째 어펜드
        if(len(max_list) == 2 ):
            max2 = max_list[1]
            a2.append(max2)
        else:
            max1 = max_list[0]
            a2.append(max1)
        

a1 = np.array(a1)
a2 = np.array(a2)
a3 = np.array(a3)
print(a1.shape, a2.shape, a3.shape) #(72000,) (72000,) (72000,)


sub = pd.read_csv('C:/data/lotte/csv/sample.csv')
sub['prediction'] = a1
sub.to_csv(f'C:/data/lotte/counter_csv/answer_add/answer5_add1_{num}.csv',index=False)
sub['prediction'] = a2
sub.to_csv(f'C:/data/lotte/counter_csv/answer_add/answer5_add2_{num}.csv',index=False)
sub['prediction'] = a3
sub.to_csv(f'C:/data/lotte/counter_csv/answer_add/answer5_add3_{num}.csv',index=False)