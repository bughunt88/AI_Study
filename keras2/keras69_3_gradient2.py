import numpy as np


f = lambda x : x**2 - 4*x + 6

gradient = lambda x : 2*x - 4

x0 = 10.0
epoch = 300
learning_rate = 0.01

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)
    x0 = temp

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))
