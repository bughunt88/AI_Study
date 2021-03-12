import tensorflow as tf
import numpy as np

#tf.set_random_seed(66)

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False

print(tf.__version__)


# 1. 데이터
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test,y_test)= cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

# (50000, 32, 32, 3)
# (50000, 1)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,32,32,3).astype('float32')/255
x_test = x_test.reshape(-1,32,32,3).astype('float32')/255

learing_rate = 0.004
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Conv2D(filter, kernel_size, input_shape)
# Conv2D(10,(3,3), input_shape(28,28,1))

w1 = tf.compat.v1.get_variable("w1", shape=[3,3,3,32]) # (커널 사이즈, 커널사이즈, 컬러, 노드의 수)
L1 = tf.nn.conv2d(x,w1, strides=[1,1,1,1], padding="SAME")
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L1)

w2 = tf.compat.v1.get_variable("w2", shape=[3,3,32,64]) # (커널 사이즈, 커널사이즈, 컬러, 노드의 수)
L2 = tf.nn.conv2d(L1,w2, strides=[1,1,1,1], padding="SAME")
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L2)

w3 = tf.compat.v1.get_variable("w3", shape=[3,3,64,128]) 
L3 = tf.nn.conv2d(L2,w3, strides=[1,1,1,1], padding="SAME")
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L3)

w4 = tf.compat.v1.get_variable("w4", shape=[3,3,128,64]) 
L4 = tf.nn.conv2d(L3,w4, strides=[1,1,1,1], padding="SAME")
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L4)

# Flaten
L_flat = tf.reshape(L4, [-1,2*2*64])
print("플래튼 : ", L_flat)

w5 = tf.compat.v1.get_variable("w5", shape=[2*2*64,64],initializer=tf.compat.v1.initializers.he_normal())
b5 = tf.Variable(tf.compat.v1.random_normal([64],name = 'b5'))
L5 = tf.nn.selu(tf.matmul(L_flat, w5)+b5)
#L5 = tf.nn.dropout(L5, keep_prob=0.2)

print(L5)

w6 = tf.compat.v1.get_variable("w6", shape=[64,32],initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.Variable(tf.compat.v1.random_normal([32],name = 'b6'))
L6 = tf.nn.selu(tf.matmul(L5, w6)+b6)
#L6 = tf.nn.dropout(L6, keep_prob=0.2)

print(L6)

w7 = tf.compat.v1.get_variable("w7", shape=[32,10],initializer=tf.compat.v1.initializers.he_normal())
b7 = tf.Variable(tf.compat.v1.random_normal([10],name = 'b7'))
hypothesis = tf.nn.softmax(tf.matmul(L6, w7)+b7)

print(hypothesis)


# 3. 컴파일, 훈련

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis = 1)) # categorical_cross_entropy
optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)

# 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict = feed_dict)
        avg_loss += c/total_batch
    
    print(f'Epoch {epoch} \t===========>\t loss : {avg_loss:.8f}')

print('훈련 끝')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))

# 훈련 끝
# Acc :  0.9691
