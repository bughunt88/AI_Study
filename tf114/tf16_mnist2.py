import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32', [None, 10])

w = tf.Variable(tf.random.normal([784, 10]), name = 'weight')
b = tf.Variable(tf.random.normal([1, 10]), name = 'bias')

#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# 컴파일 훈련(다중분류)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(2001):
        _, cur_loss = sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
        if epoch%10 == 0:
            y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
            y_pred = np.argmax(y_pred, axis = 1)
            print(f'Epoch {epoch} \t===========>\t loss : {cur_loss}')

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis = 1)

    print('accuracy score : ', accuracy_score(y_test, y_pred))

# accuracy score :  0.9738

# accuracy score :  0.9765  >> adam 0.01 / epoch 251