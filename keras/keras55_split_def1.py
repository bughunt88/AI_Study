import numpy as np
import pandas as pd

df = pd.read_csv('../data/solar/train.csv', index_col=-1,header=0,encoding='CP949')

df = df.astype('float32')

total_data = df.to_numpy()


step_x = 5
step_y =2 

def split_xy(dataset, time_steps, y_column) : 
    x, y = list(), list()                      
    for i in range(len(dataset)) :             
        x_end_number = i + time_steps         
        y_end_number = x_end_number + y_column  
        if y_end_number > len(dataset) :        
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number] 
        x.append(tmp_x)                     
        y.append(tmp_y)
    return np.array(x), np.array(y)


x, y = split_xy(total_data, step_x, step_y)

print(x,y)