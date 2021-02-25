import numpy as np
import pandas as pd
import cv2
import utils
import os

train_a = pd.read_csv('../data/vision2/mnist_data/test.csv')
train = pd.read_csv('../data/vision2/mnist_data/train.csv')

tmp1 = pd.DataFrame()

train = train.drop(['id','digit','letter'],1)
train_a = train_a.drop(['id','letter'],1)

tmp1 = pd.concat([train,train_a])

data = tmp1

print(data)

def convert2image(row):
    pixels = row['pixels']  # In dataset,row heading was 'pixels'
    img = np.array(pixels.split())
    img = img.reshape(48,48)  # dimensions of the image
    image = np.zeros((48,48,1))  # empty matrix
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return image.astype(np.uint8) # return the image


count = 0  # initialize counter
for i in range(1, data.shape[0]):  #data.shape[0] gives no. of rows
  face = data.iloc[i]  # remove one row from the data
  img = convert2image(face)  # send this row of to the function 
  # cv2.imshow("image", img) 
  # cv2.waitKey(0)  # closes the image window when you press a key
  count+=1  # counter to save the images with different name         
  cv2.imwrite(r'../data/vision2/cut/'+ str(count) +'.jpg',img) # path of location