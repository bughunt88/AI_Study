import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))