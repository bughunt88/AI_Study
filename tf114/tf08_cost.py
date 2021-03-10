import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [2.,4.,6.]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
const_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30,50):
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_history.append(curr_w)
        const_history.append(curr_cost)

print("==============")
print(w_history)
print("==============")
print(const_history)
print("==============")

plt.plot(w_history, const_history)
plt.show()