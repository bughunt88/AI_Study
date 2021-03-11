import tensorflow as tf
import numpy as np

from sklearn.datasets import load_iris

dataset = load_iris()

x_data = dataset.data
y_data = dataset.target

print(x_data.shape, y_data.shape)
# (150, 4) (150,)

y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape)
# (150, 4) (150, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, random_state = 42)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# column을 one-hot-encode
ohe = OneHotEncoder()
ohe.fit(y_train)
y_data = ohe.transform(y_train).toarray()