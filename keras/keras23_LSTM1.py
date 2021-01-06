# RNN
# 순환 신경망 
# 시계열 (time servies) 데이터에 자주 쓰인다

# 회기 모델은 그래프의 선을 그어 처리한다
# 순환 모델은 데이터의 앞에 값을 계산 후 뒤에 값을 계산한다

# LSTM을 사용하려면 x값이 3차원이여야 한다 

# (행, 열, 몇 개씩 자르는지)



# 1. 데이터

import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
# 시계열 데이터의 y는 우리가 만든다


print(x.shape) # (4,3)
print(y.shape) # (4,)

 
x = x.reshape(4,3,1) # 3차원 
# 순환 하려면 reshape 해줘야 한다 (LSTM 레이어에 넣기 위해서)

# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(3,1))) # 3차원
# 순차적으로 계산해야하기 떄문에 input_shape가 회기랑 다르다 

model.add(Dense(20)) # 2차원
model.add(Dense(10))
model.add(Dense(1))

model.summary()




# [batch, timesteps, feature]
# [batch, timesteps, input_dim]

# activation == tanh (디폴트)

# ( input_dim + binary + output) * 4 * output
# ( 1 + 1 + 10 ) * 4 * 10 = 480
# 4개의 gate가 있다 ( 4를 곱해야함 )


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x,y)
print(loss)


x_pred = np.array([5,6,7]) #(3,) -> (1,3,1)
x_pred = x_pred.reshape(1,3,1)
# lstm에 쓸 수 있는 데이터 구조로 변경 (reshape)

result = model.predict(x_pred)
print(result)

# 0.0299268439412117
# [[8.190641]]





