
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train= x_train.reshape(60000, 28*28)/255.
x_test= x_test.reshape(10000, 28*28)/255.

print(y_train)
print(y_train.shape)

#다중분류 y원핫코딩
y_train = tf.keras.utils.to_categorical(y_train)
#y_test = tf.keras.utils.to_categorical(y_test)  #(10000, 10)

#print(y_train)
print(y_train.shape)

x = tf.placeholder('float', [None, 28*28])
y = tf.placeholder('float', [None, 10])

#변수

#w1 = tf.get_variable("weight1", shape=[28*28,10], initializer=tf.contrib.layers.xavier_initializer())
w1 = tf.Variable(tf.random.normal([28*28, 10], stddev= 0.1, name = 'weight1'))
b1 = tf.Variable(tf.random_normal([1,10]), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# input 레이어

#w2 = tf.get_variable("weight2", shape=[10,7], initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.Variable(tf.random.normal([10, 7], stddev= 0.1, name = 'weight2'))
b2 = tf.Variable(tf.random_normal([7]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

#w3 = tf.get_variable("weight3", shape=[7,1], initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.Variable(tf.random.normal([7, 1], stddev= 0.1, name = 'weight3'))
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.nn.relu(tf.matmul(layer2, w3) + b3)
# hidden 레이어

#loss = tf.reduce_mean(tf.square(hypothesis-y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1)) # categorical_cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)

from sklearn.metrics import r2_score, accuracy_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train,y:y_train})
        if step % 200 == 0:
            print(step, cost_val)

    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    print(accuracy_score(y_test,y_pred ))

    