import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,28*28).astype('float32')/255.
x_test = x_test.reshape(-1,28*28).astype('float32')/255.

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])


# 모델구성

w = tf.Variable(tf.random_normal([784, 10]), name='weight1')
b = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.nn.elu(tf.matmul(x, w) + b)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3)
# 드랍아웃 방법 

w2 = tf.Variable(tf.random_normal([100,50]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)

w3 = tf.Variable(tf.random_normal([50,10]), name = 'weight2')
b3 = tf.Variable(tf.random_normal([10]), name='bias2')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)


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