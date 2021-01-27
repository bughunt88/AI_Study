from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
# 1. 데이터 
x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,11])
print("\n", x, "\n" ,y)


#2. 모델구성
model = Sequential()
model.add(Dense(1,input_shape=(1,)))
#model.add(Dense(10))
#model.add(Dense(10))
#model.add(Dense(1))


#3. 컴파일, 훈련
optimizer = RMSprop(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x,y, epochs=10)

y_pred = model.predict(x)

# 컬럼이 많으면 컬럼별로 찍어볼 것 
plt.scatter(x,y)

plt.plot(x,y_pred, color='red')
plt.show()
