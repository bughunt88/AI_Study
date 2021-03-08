from sklearn.covariance import EllipticEnvelope

import numpy as np

aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100,5000],[1000,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])
aaa = np.transpose(aaa)

outlier = EllipticEnvelope(contamination=.2)
outlier.fit(aaa)

print(outlier.predict(aaa))
