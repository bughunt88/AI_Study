# 즉시 실행 모드
#from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) # False


tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False


print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)

#sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))