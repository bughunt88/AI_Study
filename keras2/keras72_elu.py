  
import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1):
    return x if x>=0 else alpha*(np.exp(x)-1)

x = np.arange(-5,5,0.1)
print(x)
y=[]
for i in x:
    a = elu(i,1)
    y.append(a)
print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()