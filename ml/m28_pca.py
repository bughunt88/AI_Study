# PCA - 컬럼 압축!
# 통상적으로 pca.explained_variance_ratio_의 합이 0.95 이상이면 성능 비슷하다 

# feature_importances : 컬럼 특성 자체의 중요도를 찾는 것 
# PCA : 컬럼을 압축시켜 차이를 보는 방법 

# 그렇다면 feature_importances로 컬럼을 정제하고 PCA하면 좋아진다

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

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

