
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target


# 2. 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

model = LinearSVC()


# 3. 컴파일, 훈련


# model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])
model.fit(x,y)

# result = model.evaluate(x,y)
result = model.score(x,y)

print(result)

y_pred = model.predict(x[-5:-1])

print(y_pred)
print(y[-5:-1])
