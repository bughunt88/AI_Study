# PCA - 데이터 컬럼을 특성에 맞춰서 줄여주는 방법
# 통상적으로 pca.explained_variance_ratio_의 합이 0.95 이상이면 성능 비슷하다 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

'''
pca = PCA(n_components=7)
# n_components는 데이터를 지정한 수 만큼 줄여준다 

x2 = pca.fit_transform(x)
print(x2)
print(x2.shape)
# shape가 변하는 것 확인 가능 

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))
# 변화된 값을 볼 수 있다 
'''

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

print("cumsum : ",cumsum)

d = np.argmax(cumsum >= 0.95)
print("cumsum >=0.95", cumsum >= 0.95)
print("d : ", d)


import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


