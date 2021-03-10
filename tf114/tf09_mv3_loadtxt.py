import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

dataset = np.loadtxt("../data/naver/data-01-test-score.csv",delimiter=',')

xy_pred = dataset[:5]
xy_train = dataset[5:]

x_pred = xy_pred[:,:-1]
y_real = xy_pred[:,-1].reshape(-1,1)
x_train = xy_train[:,:-1]
y_train = xy_train[:,-1].reshape(-1,1)

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

'''
print(x_data.shape) #(25, 3)
print(y_data.shape) #(25,)
y_data = y_data.reshape(25, 1)
'''

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b # 일반 연산
hypothesis = tf.matmul(x, w) + b # 행렬 연산 

loss = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate= 0.000001).minimize(loss)
#train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost) # optimizer + train

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(10001):
        cur_loss, cur_hypo, _ = sess.run([loss, hypothesis, train], feed_dict = {x:x_train, y:y_train})
        if epoch%20 == 0:
            print(f'Epoch {epoch} loss : {cur_loss}')
    print("predict 값 : ",sess.run(hypothesis, feed_dict={x:x_pred}))

print("정답 : ", y_real)

