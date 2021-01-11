import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test,y_test)= cifar100.load_data()


print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.max(y_test))
print(np.max(y_train))
# y 범위 0 ~ 9


print(np.max(x_test))
print(np.max(x_train))

print(x_train[0].shape) # (28,28)

# plt.imshow(x_train[1])
plt.show()