import tensorflow as tf
import numpy as np

from sklearn.datasets import load_iris

dataset = load_iris()
x_data = dataset.data
y_data = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train)
labels = encoder.transform(y_train)

# 2차원 데이터로 변환합니다. 
labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
y_train = oh_encoder.transform(labels).toarray()


#(150, 4)
#(150, 3)

x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float', [None, 3])

w = tf.Variable(tf.random_normal([4,3]))
b = tf.Variable(tf.random_normal([1,3]))

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

#loss = tf.reduce_mean(tf.square(hypothesis-y)) # mse

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1)) # categorical_cross_entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

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

    