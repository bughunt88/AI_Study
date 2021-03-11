import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32', [None, 10])

#2. 모델구성
w1 = tf.get_variable('w1', shape = [784, 100],
                    initializer= tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random.normal([1, 100]), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob= 0.3)

w2 = tf.get_variable('weight2', shape = [100, 128],
                    initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([1, 128]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
# layer2 = tf.nn.dropout(layer2, keep_prob= 0.3)

w3 = tf.get_variable('weight3', shape = [128, 64],
                    initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([1, 64]), name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.dropout(layer3, keep_prob= 0.3)

w4 = tf.get_variable('weight4', shape = [64, 10],
                    initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([1, 10]), name = 'bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)


# 컴파일 훈련(다중분류)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

training_epochs = 300
batch_size = 128
total_batch = int(len(x_train)/batch_size) # 60000 / 100 = 600

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, train], feed_dict = feed_dict)
        avg_loss += c/total_batch
    
    print(f'Epoch {epoch} \t===========>\t loss : {avg_loss:.8f}')

print('훈련 끝')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))

# 훈련 끝
# Acc :  0.9691