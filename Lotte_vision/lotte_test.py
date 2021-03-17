import cv2
import numpy as np
import matplotlib.pyplot as plt



x_train = np.load('../data/lpd_competition/npy/train_data_x.npy')
y_train = np.load('../data/lpd_competition/npy/train_data_y.npy')


print(x_train.shape)
print(y_train.shape)
