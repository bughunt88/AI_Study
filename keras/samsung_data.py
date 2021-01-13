import numpy as np
import pandas as pd

df = pd.read_csv('../Study/x.csv', index_col=0,header=0,encoding='CP949')

# 10, 13

df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')
df = df.sort_values(by=['일자'], axis=0)


# # 상관 계수 시각화!
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)


# plt.show()

df_x = df.loc['2018-05-04':'2021-01-12']
df_y = df.loc['2018-05-08':'2021-01-13']

df_x = df_x.iloc[:,[0,1,2,9,12]]
# (661,6)

df_y = df_y.iloc[:,[3]]

# df.replace(',', '', inplace=True)

x_data = df_x.to_numpy()
y_data = df_y.to_numpy()


# np.save('../data/npy/x_data.npy', arr=x_data)
# np.save('../data/npy/y_data.npy', arr=y_data)














# # 상관 계수 시각화!
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()




# import numpy as np

# a = np.array([range(6,11),range(1,6),range(6,11),range(1,6)])  # 1~10 
# size = 3

# def split_x(seq, size):
    
#     print(range(len(seq) - size + 1))
    
#     aaa = []
#     for i in range(len(seq) - size + 1):
#         for n in size:
#             subset = seq[i : (i+size)]
       
#         aaa.append(subset)
#     return np.array(aaa)

# dataset = split_x(a,size)
# # print("===================")
# print(dataset)
