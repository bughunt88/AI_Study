import numpy as np
import matplotlib.pyplot as plt

def relu(x) :
    return np.maximum(0,x)

x = np.arange(-5,5, 0.1)
y = relu(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()

### 과제
# elu, selu, reaky relu
# 72_2, 72_3, 72_4로 만들어바라