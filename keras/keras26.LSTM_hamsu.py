# keras23_LSTM3_scale 을 함수형으로 코딩



import numpy as np


# 1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])


y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_pred = np.array([50,60,70])

x = x.reshape(13,3,1) # 3차원 


# 2.모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM,Input



input1 = Input(shape=(3,1))
# 인풋 레이어 직접 구성
dense1 = LSTM(10, activation='relu')(input1)
# 위에서 지정한 변수 명을 아래에 써줘야 한다
dense1 = Dense(50, activation='relu')(dense1) 
dense1 = Dense(50, activation='relu')(dense1) 
dense1 = Dense(50, activation='relu')(dense1)   
dense1 = Dense(50, activation='relu')(dense1) 
dense1 = Dense(50, activation='relu')(dense1) 
dense1 = Dense(25)(dense1) 
dense1 = Dense(25)(dense1) 
dense1 = Dense(25)(dense1) 
dense1 = Dense(25)(dense1) 
dense1 = Dense(25)(dense1) 
dense1 = Dense(10)(dense1) 
dense1 = Dense(5)(dense1) 
outputs = Dense(1)(dense1)

model = Model(inputs = input1, outputs = outputs)
# 함수형 모델을 지정하려면 인풋과 아웃풋 


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=200, batch_size=1)


# 4. 평가, 예측

loss = model.evaluate(x,y)
print(loss)

x_pred = x_pred.reshape(1,3,1)
# lstm에 쓸 수 있는 데이터 구조로 변경 (reshape)

result = model.predict(x_pred)
print(result)

# keras23_LSTM3_scale 결과
# 0.10277651250362396
# [[80.14559]]



# keras26.LSTM_hansu 결과
# 0.008856565691530704
# [[79.84822]]