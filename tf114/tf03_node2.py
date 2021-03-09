# [실습]
# 더하기, 빼기, 곱하기, 나누기
# 만들 것

import tensorflow as tf

sess = tf.Session()

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)

print('더하기 : ', sess.run(node3))

node4 = tf.subtract(node1, node2)
print('빼기 : ', sess.run(node4))

node5 = tf.multiply(node1, node2)
print('곱하기 : ', sess.run(node5))

node6 = tf.divide(node1, node2)
print('나누기 : ', sess.run(node6))

node6_1 = tf.truediv(node1, node2)
print('나누기_1 : ', sess.run(node6_1))

node7 = tf.math.mod(node1, node2)
print('나누기 나머지 : ', sess.run(node7))



print(2%3)
