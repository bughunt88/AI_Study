import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)

sess = tf.Session()
print(sess.run(hello))