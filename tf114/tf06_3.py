#[실습]
# epoch가 1000번까지 줄이자 !

import tensorflow as tf

tf.set_random_seed(66)

#x_train = [1,2,3]
#y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

# print(sess.run(W),sess.run(b))

hypothesis = x_train*W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.174139).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(101):
        cost_val,W_val,b_val,_ =sess.run([cost,W,b,train], feed_dict={x_train:[1,2,3],y_train:[3,5,7]})
        if step %20 == 0:
            print(step, cost_val, W_val, b_val) # epoch, loss, weight, bias

    print("predict 값 : ",sess.run(hypothesis, feed_dict={x_train:[3]}))

