import numpy as np



# 1. 데이터 

a = np.array(range(1,11))
size = 5


def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)

# 행열 슬라이싱 
# ( 행 , 열 ) 
x = dataset[   : ,  :size-1]
y = dataset[   : ,   size-1]

x_pred = dataset[5,1:5]

# 슬라이싱 

x = x.reshape(x.shape[0],x.shape[1],1) # 3차원 

x_pred = x_pred.reshape(1,x_pred.shape[0],1) # 3차원 




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM,Input



# 2. 모델 구성 

input1 = Input(shape=(4,1))
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

result = model.predict(x_pred)
print(result)


# keras32_split1_LSTM 결과 

# 0.0005809169379062951
# [[11.1557865]]