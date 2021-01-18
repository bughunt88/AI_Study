import numpy as np
import pandas as pd



df = pd.read_csv('../data/solar/train.csv', index_col=-1,header=0,encoding='CP949')

df = df.astype('float32')

total_data = df.to_numpy()

step_x = 3
step_y =2 
feature_x = 2
feature_y = 2

def split_xy(dataset, x_row, x_col, y_row, y_col ) :
      x, y = list(), list()
      for i in range(len(dataset)) :
            x_start_number = i               
            x_end_number = i + x_row        
            y_start_number = x_end_number    
            y_end_number = y_start_number + y_row 
            if y_end_number > len(dataset) :
                  break
            tmp_x = dataset[x_start_number : x_end_number, : x_col] 
            tmp_y = dataset[y_start_number : y_end_number, x_col : x_col + y_col]
            x.append(tmp_x)
            y.append(tmp_y)
      return np.array(x), np.array(y)

x, y = split_xy(total_data, step_x, step_y, feature_x, feature_y)

print(x.shape)
print(y.shape)
