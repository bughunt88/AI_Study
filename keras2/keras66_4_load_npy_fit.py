import numpy as np

x_train = np.load('../data/image/brain/npy/kerass66_train_x.npy')
y_train = np.load('../data/image/brain/npy/kerass66_train_y.npy')
x_test = np.load('../data/image/brain/npy/kerass66_test_x.npy')
y_test = np.load('../data/image/brain/npy/kerass66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
