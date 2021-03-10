
#accuracy score
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)
# (569, 30)
# (569, 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

x= tf.placeholder(tf.float32, shape=[None,30])
y= tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.zeros([30,1]),name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  #binary_crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate=0.00000117)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val,hy_val ,_ = sess.run([cost, hypothesis,train], feed_dict={x:x_train, y:y_train})

        if step % 50 == 0:
            print(f'step : {step} \ncost : {cost_val} \nhy_val : \n{hy_val}')


    h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_test, y:y_test})

    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')


#accuracy: : 0.9210526347160339