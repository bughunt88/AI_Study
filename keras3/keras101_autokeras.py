import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = ak.ImageClassifier(
#overwrite=True,
max_trials=2
)

model.fit(x_train, y_train, epochs=3)

results = model.evaluate(x_test,y_test)

print(results)