# 회귀

from sklearn.datasets import load_diabetes
import tensorflow as tf


dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target

y_data = y_data.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)

#(442, 10)
#(442, 1)

print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([10,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


# hypothesis = x * w + b # 일반 연산
hypothesis = tf.matmul(x, w) + b # 행렬 연산 

loss = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.AdamOptimizer(learning_rate= 0.8).minimize(loss)
#train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost) # optimizer + train
from sklearn.metrics import r2_score


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(2000):
        _, cur_loss, cur_hypothesis, cur_w, cur_b = sess.run([train, loss, hypothesis, w, b], feed_dict= {x:x_train, y:y_train})
        if epoch%20 == 0:
            print(f'Epoch : {epoch} >>> loss : {cur_loss}\nhypo : {cur_hypothesis}')
    y_predict1 = sess.run(hypothesis, feed_dict={x:x_test})
    R2 = r2_score(y_test, y_predict1)
    print('R2: ', R2)   
