import tensorflow as tf

tf.set_random_seed(66)
x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

# print(sess.run(W),sess.run(b))

hypothesis = x_train*W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step<=2:
        print("epoch : ",step)
        print("x : ",x_train)
        print("W : ",sess.run(W))
        print("b : ",sess.run(b))
        print("W*x + b = hypothesis , ",sess.run(hypothesis))
        print("y : ",y_train)
        print("hypothesis - y_train : ",sess.run(hypothesis - y_train))
        print("cost : ",sess.run(cost))
        print("\n\n")