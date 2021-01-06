# 과제 및 실습 Dense
# 지금까지 배운 모든 것 다 넣기 

# 데이터 1~100 /
#        x            y
# 1, 2, 3, 4, 5       6
#     ...
# 95,96,97,98,99     100

# predict를 만들 것
# 96, 97, 98, 99, 100 -> 101
#      ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101,102,103,104,105)



# LSTM 과 결과 비교!!!




import numpy as np



a = np.array(range(1,101))

b = np.array(range(96,105))

size = 10


def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)

x_predict = split_x(b,size-1)

# 행열 슬라이싱 
# ( 행 , 열 ) 
x = dataset[   : ,  :size-1]
y = dataset[   : ,   size-1]

# 슬라이싱 


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM



# 2. 모델 구성 

input1 = Input(shape=(size-1,))
dense1 = Dense(10, activation='relu')(input1)
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


# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')




from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


# 4. 평가, 예측



loss = model.evaluate(x_test, y_test, batch_size=16)


print(loss)

y_predict = model.predict(x_predict)
# 지표를 만들기 위한 프레딕트 

print(y_predict)



# keras32_split3_LSTM 결과

# 0.007421054877340794
# [[101.101906]
#  [102.14314 ]
#  [103.18791 ]
#  [104.23622 ]


# keras32_split4_Dense
# 4.9763431889005005e-05
# [[101.009705]
# [102.00993 ]
# [103.01018 ]
# [104.0104  ]
# [105.01063 ]]