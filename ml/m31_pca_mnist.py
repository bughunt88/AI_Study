import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis = 0)

print(x.shape) # 70000,28,28

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

print(x.shape)


pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

print("cumsum : ",cumsum)

d = np.argmax(cumsum >= 0.95)
print("cumsum >=0.95", cumsum >= 0.95)
print("d : ", d)


# 0.95 이상 153개 


import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
