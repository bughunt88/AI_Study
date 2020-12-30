# 행렬이 2차원 일 때 input_dim 에 넣는 값도 달라진다 (컬럼의 수로 결정한다)

# 다:1 mlp

import numpy as np

# 1. 데이터 

x = np.array( [[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]  )   # 지금은 (2,10) 이다 

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)


# input_dim 은 컬럼의 수를 넣는다

# 스크린 샷 찍어 둠 행열 문제 
# 1. 2행 3열
# 2. 3행 2열
# 3. 1, 2행 3열 
# 4. 1행 6열 
# 5. 2, 2행 2열
# 6. 3행 1열
# 7. 2, 2행 1열 

# 행렬은 가장 작은 단위부터 확인한다 가장 작은 단위는 열이다(컬럼)


# 위에 (2,10)을 (10,2)로 변경해주는 함수
# x = x.reshape((10, 2)) 
x = np.transpose(x)

print(x)
print(x.shape)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim= 2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가, 예측

loss, mae = model.evaluate(x,y)

print('loss : ', loss)
print('mae : ', mae)


y_predict = model.predict(x)

print(y_predict)


'''
# 사이킷런
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
    # mean_squared_error는 sklearn에서 mse 만드는 함수 
    # sqrt는 넘파이에 루트 씌우는 함수



print('loss : ', loss)
print('mae : ', mae )


print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))



# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''
