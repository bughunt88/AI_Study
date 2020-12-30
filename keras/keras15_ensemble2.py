
# 실습 다:1 앙상블을 구현하시오


# ㅁ   ㅁ
#   ㅁ
#   ㅁ

# 위와 같은 모델 형식

import numpy as np

# 1. 데이터

x1 = np.array([ range(100), range(301,401), range(1,101)  ])   # 지금은 (3,100) 이다 
x2 = np.array([range(101,201), range(411,511), range(100,200)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])
# y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test, x2_train, x2_test = train_test_split(x1, y1, x2, shuffle=False, train_size=0.8) # shuffle False면 섞는다



# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Input
# 텐서플로우 레이어에 여러가지  

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers from concatenate, Concatenate



merge1 = concatenate([dense1, dense2])
# 모델 1과 모델 2를 엮는 코드

middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(3)(middle1)
# 엮은 후 레이어 구성도 가능하다

# 모델 선언
model = Model(inputs=[input1,input2], outputs=[middle1])

model.summary()



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train,x2_train], [y1_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)
# *데이터 2개 이상은 무조건 리스트로 묶는다


# 4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], [y1_test], batch_size=1 )

print(loss)

print("model.metrics_names : ", model.metrics_names)
# 모델의 메트릭스 네임을 확인할 수 있는 코드


y1_predict = model.predict([x1_test, x2_test])

print(y1_predict)



# 사이킷런
from sklearn.metrics import mean_squared_error

# RMSE 
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
    # mean_squared_error는 sklearn에서 mse 만드는 함수 
    # sqrt는 넘파이에 루트 씌우는 함

print("rmse : ", RMSE(y1_test,y1_predict))


# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y1_test , y1_predict)

print("r2 : ", r2)
